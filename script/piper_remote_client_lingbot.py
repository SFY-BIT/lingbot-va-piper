import argparse
import csv
import logging
import pathlib
import sys
import threading
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

LINGBOT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(LINGBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT))

from evaluation.robotwin import websocket_client_policy

DEFAULT_LEROBOT_ROOT = pathlib.Path("/mnt/hdd/sfy/lerobot.act")
DEFAULT_BASE_IMAGE_KEY = "observation.images.one"
DEFAULT_WRIST_IMAGE_KEY = "observation.images.two"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_POLICY_BASE_IMAGE_KEY = "observation.images.one"
DEFAULT_POLICY_WRIST_IMAGE_KEY = "observation.images.two"
DEFAULT_MASKED_JOINT_INDEX = 3


def _load_make_robot(lerobot_root: pathlib.Path):
    if not lerobot_root.exists():
        raise FileNotFoundError(f"LeRobot root does not exist: {lerobot_root}")

    sys.path.insert(0, str(lerobot_root))

    try:
        from lerobot.common.robot_devices.robots.utils import make_robot
    except ImportError as exc:
        raise ImportError(
            "Failed to import Piper runtime from lerobot.act. "
            f"Make sure {lerobot_root} is the repo root and its dependencies are installed."
        ) from exc

    return make_robot


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _convert_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    if image.max() <= 1.0 and image.min() >= 0.0:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0)
    return image.astype(np.uint8)


def _resize_with_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {image.shape}")

    image = _convert_to_uint8(image)
    in_h, in_w = image.shape[:2]
    if in_h <= 0 or in_w <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = min(target_w / in_w, target_h / in_h)
    new_w = max(1, int(round(in_w * scale)))
    new_h = max(1, int(round(in_h * scale)))

    pil = Image.fromarray(image)
    resized = np.asarray(pil.resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.uint8)

    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def _prepare_image(value: Any, image_size: int) -> np.ndarray:
    return _resize_with_pad(_to_numpy(value), image_size, image_size)


def _extract_action_chunk(action_chunk: Any, expected_state_dim: int) -> np.ndarray:
    action_chunk = _to_numpy(action_chunk).astype(np.float32)

    if action_chunk.ndim == 1:
        return action_chunk[None, :]

    if action_chunk.ndim == 2:
        if expected_state_dim > 0:
            if action_chunk.shape[0] == expected_state_dim:
                # [C, T] -> [T, C]
                return action_chunk.T
            if action_chunk.shape[1] == expected_state_dim:
                # [T, C]
                return action_chunk
        # Fallback to treating first axis as time.
        return action_chunk

    if action_chunk.ndim == 3:
        # Lingbot-VA returns [C, F, H] for chunked actions.
        if expected_state_dim > 0 and action_chunk.shape[0] != expected_state_dim:
            raise ValueError(
                "Expected lingbot action chunk shape [C,F,H] with "
                f"C={expected_state_dim}, got {action_chunk.shape}"
            )
        c_dim, f_dim, h_dim = action_chunk.shape
        return np.transpose(action_chunk, (1, 2, 0)).reshape(f_dim * h_dim, c_dim)

    raise ValueError(f"Unsupported action chunk shape {action_chunk.shape}; expected rank 1/2/3")


def _read_current_state(robot: Any, expected_state_dim: int) -> np.ndarray:
    if not hasattr(robot, "arm") or not hasattr(robot.arm, "read"):
        raise AttributeError("Piper remote client expects the robot runtime to expose robot.arm.read().")

    state = np.asarray(list(robot.arm.read().values()), dtype=np.float32)
    if expected_state_dim > 0 and state.shape[0] != expected_state_dim:
        raise ValueError(f"Expected {expected_state_dim}-D robot state, got {state.shape[0]}")
    return state


def _clamp_action_delta(
    target_action: np.ndarray,
    current_state: np.ndarray,
    *,
    max_abs_joint_delta: float | None,
    max_abs_gripper_delta: float | None,
) -> tuple[np.ndarray, bool]:
    if target_action.shape != current_state.shape:
        raise ValueError(
            f"Target action shape {target_action.shape} does not match current state shape {current_state.shape}"
        )

    clamped = np.array(target_action, copy=True)
    was_clamped = False
    gripper_index = clamped.shape[0] - 1

    for idx, (target_value, current_value) in enumerate(zip(target_action, current_state, strict=True)):
        max_delta = max_abs_gripper_delta if idx == gripper_index else max_abs_joint_delta
        if max_delta is None or max_delta <= 0:
            continue

        bounded = np.clip(float(target_value), float(current_value - max_delta), float(current_value + max_delta))
        if bounded != float(target_value):
            was_clamped = True
        clamped[idx] = bounded

    return clamped, was_clamped


def _low_pass_action(
    target_action: np.ndarray,
    previous_action: np.ndarray | None,
    *,
    alpha: float,
) -> tuple[np.ndarray, bool]:
    if previous_action is None or alpha >= 1.0:
        return np.array(target_action, copy=True), False
    if alpha <= 0.0:
        return np.array(previous_action, copy=True), not np.allclose(previous_action, target_action)
    if target_action.shape != previous_action.shape:
        raise ValueError(
            f"Target action shape {target_action.shape} does not match previous action shape {previous_action.shape}"
        )

    filtered = ((1.0 - alpha) * previous_action) + (alpha * target_action)
    filtered = filtered.astype(np.float32, copy=False)
    return filtered, not np.allclose(filtered, target_action)


def _suppress_small_action_update(
    target_action: np.ndarray,
    previous_action: np.ndarray | None,
    *,
    min_abs_joint_change: float | None,
    min_abs_gripper_change: float | None,
) -> tuple[np.ndarray, bool]:
    if previous_action is None:
        return np.array(target_action, copy=True), False
    if target_action.shape != previous_action.shape:
        raise ValueError(
            f"Target action shape {target_action.shape} does not match previous action shape {previous_action.shape}"
        )

    gripper_index = target_action.shape[0] - 1
    for idx, (target_value, previous_value) in enumerate(zip(target_action, previous_action, strict=True)):
        min_delta = min_abs_gripper_change if idx == gripper_index else min_abs_joint_change
        min_delta = 0.0 if min_delta is None else float(min_delta)
        if abs(float(target_value) - float(previous_value)) > min_delta:
            return np.array(target_action, copy=True), False

    return np.array(previous_action, copy=True), True


def _build_policy_observation(
    observation: dict[str, Any],
    task: str,
    image_size: int,
    expected_state_dim: int,
    base_image_key: str,
    wrist_image_key: str,
    state_key: str,
    policy_base_image_key: str,
    policy_wrist_image_key: str,
) -> tuple[dict[str, Any], np.ndarray]:
    missing_keys = [
        key
        for key in (base_image_key, wrist_image_key, state_key)
        if key not in observation
    ]
    if missing_keys:
        raise KeyError(f"Observation is missing required keys: {missing_keys}")

    state = _to_numpy(observation[state_key]).astype(np.float32)
    if state.ndim != 1:
        raise ValueError(f"Expected 1-D state, got shape {state.shape}")
    if expected_state_dim > 0 and state.shape[0] != expected_state_dim:
        raise ValueError(f"Expected {expected_state_dim}-D state for Piper control, got {state.shape[0]}")

    policy_observation = {
        "obs": [
            {
                policy_base_image_key: _prepare_image(observation[base_image_key], image_size),
                policy_wrist_image_key: _prepare_image(observation[wrist_image_key], image_size),
            }
        ],
        "prompt": task,
    }
    return policy_observation, state


def _resolve_loop_timing(args: argparse.Namespace) -> tuple[float | None, float | None]:
    if args.policy_fps is not None:
        if args.policy_fps <= 0:
            raise ValueError(f"--policy-fps must be > 0, got {args.policy_fps}")
        if args.chunk_execute_steps <= 0:
            raise ValueError("--chunk-execute-steps must be > 0 when --policy-fps is set.")
        return float(args.policy_fps), float(args.chunk_execute_steps) * float(args.policy_fps)

    if args.fps is not None and args.fps <= 0:
        raise ValueError(f"--fps must be > 0 when provided, got {args.fps}")
    return None, None if args.fps is None else float(args.fps)


def _fetch_chunk_from_policy(
    policy: Any,
    policy_observation: dict[str, Any],
    *,
    action_key: str,
    expected_state_dim: int,
    chunk_execute_steps: int,
) -> dict[str, Any]:
    policy_start_t = time.perf_counter()
    policy_result = policy.infer(policy_observation)
    infer_recv_t = time.perf_counter()
    policy_roundtrip_ms = (infer_recv_t - policy_start_t) * 1000

    if action_key not in policy_result:
        raise KeyError(
            f"Policy response is missing '{action_key}'. Available keys: {sorted(policy_result.keys())}"
        )

    extract_start_t = time.perf_counter()
    pending_actions = _extract_action_chunk(policy_result[action_key], expected_state_dim)
    raw_chunk_len = len(pending_actions)
    if chunk_execute_steps > 0:
        pending_actions = pending_actions[:chunk_execute_steps]
    executed_chunk_len = len(pending_actions)
    chunk_extract_ms = (time.perf_counter() - extract_start_t) * 1000

    return {
        "pending_actions": pending_actions,
        "raw_chunk_len": raw_chunk_len,
        "executed_chunk_len": executed_chunk_len,
        "policy_timing": policy_result.get("policy_timing", {}),
        "server_timing": policy_result.get("server_timing", {}),
        "policy_roundtrip_ms": policy_roundtrip_ms,
        "chunk_extract_ms": chunk_extract_ms,
        "infer_recv_t": infer_recv_t,
    }


def _prefetch_worker(
    *,
    policy: Any,
    policy_observation: dict[str, Any],
    action_key: str,
    expected_state_dim: int,
    chunk_execute_steps: int,
    output: dict[str, Any],
) -> None:
    try:
        output["result"] = _fetch_chunk_from_policy(
            policy,
            policy_observation,
            action_key=action_key,
            expected_state_dim=expected_state_dim,
            chunk_execute_steps=chunk_execute_steps,
        )
    except Exception as exc:
        output["error"] = exc


def _make_trace_fieldnames(state_dim: int) -> list[str]:
    fieldnames = [
        "step_idx",
        "wall_time_s",
        "monotonic_s",
        "t_rel_s",
        "inter_step_ms",
        "infer_seq",
        "infer_requested",
        "prefetch_started",
        "prefetch_used",
        "prefetch_wait_ms",
        "infer_interval_ms",
        "action_age_ms",
        "chunk_step",
        "chunk_size",
        "action_suppressed",
        "was_clamped",
        "was_filtered",
        "was_reclamped",
        "capture_ms",
        "prepare_ms",
        "read_state_ms",
        "policy_roundtrip_ms",
        "chunk_extract_ms",
        "clamp_ms",
        "filter_ms",
        "action_send_ms",
        "loop_compute_ms",
        "cycle_ms",
        "sleep_ms",
        "effective_hz",
        "network_overhead_ms",
        "server_infer_ms",
        "server_total_ms",
        "policy_total_ms",
    ]
    for prefix in ("current_state", "raw_action", "final_action", "executed_action"):
        for idx in range(state_dim):
            fieldnames.append(f"{prefix}_{idx}")
    return fieldnames


def _write_trace_vector(row: dict[str, Any], prefix: str, values: np.ndarray | None, state_dim: int) -> None:
    for idx in range(state_dim):
        key = f"{prefix}_{idx}"
        if values is None or idx >= len(values):
            row[key] = None
        else:
            row[key] = float(values[idx])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Piper robot locally and query a Lingbot-VA websocket policy server remotely."
    )
    parser.add_argument(
        "--lerobot-root",
        type=pathlib.Path,
        default=DEFAULT_LEROBOT_ROOT,
        help="Path to the local lerobot.act repository that contains the Piper robot runtime.",
    )
    parser.add_argument("--server-host", default="127.0.0.1", help="Hostname or IP of the Lingbot-VA policy server.")
    parser.add_argument("--server-port", type=int, default=8000, help="Port of the Lingbot-VA policy server.")
    parser.add_argument("--task", required=True, help="Task prompt to send to the policy.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Requested local control loop frequency. "
            "If omitted, use --policy-fps together with --chunk-execute-steps to derive it automatically."
        ),
    )
    parser.add_argument(
        "--policy-fps",
        type=float,
        default=None,
        help=(
            "Target frequency for requesting fresh action chunks from the policy server. "
            "Requires --chunk-execute-steps > 0 and derives local execution fps automatically."
        ),
    )
    parser.add_argument("--duration-s", type=float, default=60.0, help="How long to run the control loop.")
    parser.add_argument("--robot-type", default="piper", help="Robot type exposed by lerobot.act.")
    parser.add_argument(
        "--chunk-execute-steps",
        type=int,
        default=1,
        help="Maximum number of actions to execute from each returned action chunk before refreshing from the server.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize both camera images to this square size before sending them to the server.",
    )
    parser.add_argument(
        "--expected-state-dim",
        type=int,
        default=7,
        help="Expected robot state dimension. Lingbot piper config in this repo uses 7.",
    )
    parser.add_argument(
        "--obs-base-image-key",
        default=DEFAULT_BASE_IMAGE_KEY,
        help="Camera key from robot.capture_observation() used as the first policy image.",
    )
    parser.add_argument(
        "--obs-wrist-image-key",
        default=DEFAULT_WRIST_IMAGE_KEY,
        help="Camera key from robot.capture_observation() used as the second policy image.",
    )
    parser.add_argument(
        "--obs-state-key",
        default=DEFAULT_STATE_KEY,
        help="State key from robot.capture_observation().",
    )
    parser.add_argument(
        "--policy-base-image-key",
        default=DEFAULT_POLICY_BASE_IMAGE_KEY,
        help="Image key name expected by Lingbot server for the first camera.",
    )
    parser.add_argument(
        "--policy-wrist-image-key",
        default=DEFAULT_POLICY_WRIST_IMAGE_KEY,
        help="Image key name expected by Lingbot server for the second camera.",
    )
    parser.add_argument(
        "--policy-action-key",
        default="action",
        help="Action key in Lingbot server response.",
    )
    parser.add_argument(
        "--skip-server-reset",
        action="store_true",
        help="Do not send a reset request to server at startup.",
    )
    parser.add_argument(
        "--hold-joint-index",
        type=int,
        default=DEFAULT_MASKED_JOINT_INDEX,
        help=(
            "Action index to overwrite with the current joint value before sending to the robot. "
            "Defaults to 3 for joint3mask checkpoints."
        ),
    )
    parser.add_argument(
        "--max-abs-joint-delta",
        type=float,
        default=0.2,
        help="Maximum per-step absolute change for each arm joint target relative to the current state.",
    )
    parser.add_argument(
        "--max-abs-gripper-delta",
        type=float,
        default=0.02,
        help="Maximum per-step absolute change for the gripper target relative to the current state.",
    )
    parser.add_argument(
        "--action-filter-alpha",
        type=float,
        default=0.35,
        help=(
            "Low-pass smoothing weight applied to the final robot command. "
            "Use a value in (0, 1]; smaller values are smoother, 1 disables filtering."
        ),
    )
    parser.add_argument(
        "--min-abs-joint-action-change",
        type=float,
        default=0.002,
        help=(
            "Skip sending the action if every non-gripper joint changes by no more than this amount "
            "relative to the previously executed action. Set to 0 to disable joint deadband."
        ),
    )
    parser.add_argument(
        "--min-abs-gripper-action-change",
        type=float,
        default=0.002,
        help=(
            "Skip sending the action if the gripper change is no more than this amount "
            "relative to the previously executed action. Set to 0 to disable gripper deadband."
        ),
    )
    parser.add_argument(
        "--trace-csv-path",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional CSV path for per-step telemetry. "
            "Includes timestamps, joint states, commanded actions, and infer-to-execution timing."
        ),
    )
    parser.add_argument(
        "--prefetch-trigger-step",
        type=int,
        default=5,
        help=(
            "Start prefetching the next chunk when executing this step index (1-based) within the current chunk. "
            "Set to 0 to disable prefetch."
        ),
    )
    args = parser.parse_args()
    if args.fps is None and args.policy_fps is None:
        args.fps = 5.0
    return args


def main() -> None:
    args = _parse_args()
    policy_request_fps, control_fps = _resolve_loop_timing(args)
    if not 0.0 <= args.action_filter_alpha <= 1.0:
        raise ValueError(f"--action-filter-alpha must be in [0, 1], got {args.action_filter_alpha}")
    if args.min_abs_joint_action_change < 0.0:
        raise ValueError(
            f"--min-abs-joint-action-change must be >= 0, got {args.min_abs_joint_action_change}"
        )
    if args.min_abs_gripper_action_change < 0.0:
        raise ValueError(
            f"--min-abs-gripper-action-change must be >= 0, got {args.min_abs_gripper_action_change}"
        )
    if args.prefetch_trigger_step < 0:
        raise ValueError(f"--prefetch-trigger-step must be >= 0, got {args.prefetch_trigger_step}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    make_robot = _load_make_robot(args.lerobot_root.resolve())
    robot = make_robot(args.robot_type, inference_time=True)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    prefetch_policy = websocket_client_policy.WebsocketClientPolicy(host=args.server_host, port=args.server_port)

    logging.info("Connected to policy server: %s", policy.get_server_metadata())
    if not args.skip_server_reset:
        reset_payload = {"reset": True, "prompt": args.task}
        policy.infer(reset_payload)
        prefetch_policy.infer(reset_payload)
        logging.info("Sent reset to Lingbot server with prompt.")

    logging.info(
        "Applying joint3mask execution rule: action[%s] will be replaced with the current joint value.",
        args.hold_joint_index,
    )
    logging.info(
        "Action chunks will be executed sequentially before the next server inference request (max %s steps).",
        args.chunk_execute_steps,
    )
    if policy_request_fps is not None:
        logging.info(
            "Using derived timing: policy_request_fps=%.3fHz chunk_execute_steps=%s -> control_fps=%.3fHz.",
            policy_request_fps,
            args.chunk_execute_steps,
            control_fps,
        )
    else:
        logging.info("Using direct local control frequency: control_fps=%.3fHz.", control_fps)
    if args.action_filter_alpha < 1.0:
        logging.info(
            "Low-pass action filter enabled with alpha=%.3f (smaller is smoother, 1 disables filtering).",
            args.action_filter_alpha,
        )
    else:
        logging.info("Low-pass action filter disabled (alpha=%.3f).", args.action_filter_alpha)
    logging.info(
        "Small-action suppression enabled with joint_deadband=%.4f gripper_deadband=%.4f.",
        args.min_abs_joint_action_change,
        args.min_abs_gripper_action_change,
    )
    logging.info(
        "Chunk prefetch trigger step=%s (0 disables). Prefetch only sends early; chunk switch still happens at boundary.",
        args.prefetch_trigger_step,
    )
    if args.trace_csv_path is not None:
        logging.info("Per-step trace CSV enabled: %s", args.trace_csv_path)

    trace_file = None
    trace_writer = None
    trace_state_dim = None
    robot.connect()
    try:
        start_t = time.perf_counter()
        step_idx = 0
        pending_actions: np.ndarray | None = None
        pending_action_index = 0
        latest_policy_timing: dict[str, Any] = {}
        latest_server_timing: dict[str, Any] = {}
        raw_chunk_len = 0
        executed_chunk_len = 0
        previous_executed_action: np.ndarray | None = None
        infer_seq = -1
        latest_infer_recv_t: float | None = None
        prev_infer_recv_t: float | None = None
        prev_loop_start_t: float | None = None
        prefetch_thread: threading.Thread | None = None
        prefetch_output: dict[str, Any] | None = None
        prefetch_started_chunk_idx = -1
        chunk_fetch_count = 0
        while time.perf_counter() - start_t < args.duration_s:
            loop_start_t = time.perf_counter()
            infer_requested = False
            prefetch_started = False
            prefetch_used = False
            prefetch_wait_ms = 0.0
            capture_ms = 0.0
            prepare_ms = 0.0
            read_state_ms = 0.0
            policy_roundtrip_ms = 0.0
            chunk_extract_ms = 0.0
            action_send_ms = 0.0
            clamp_ms = 0.0
            filter_ms = 0.0
            action_suppressed = False
            was_clamped = False
            was_filtered = False
            was_reclamped = False

            if pending_actions is None or pending_action_index >= len(pending_actions):
                chunk_result: dict[str, Any]
                if prefetch_thread is not None and prefetch_output is not None:
                    wait_start_t = time.perf_counter()
                    prefetch_thread.join()
                    prefetch_wait_ms = (time.perf_counter() - wait_start_t) * 1000
                    prefetch_thread = None
                    if "error" in prefetch_output:
                        raise RuntimeError("Prefetch inference failed.") from prefetch_output["error"]
                    chunk_result = prefetch_output["result"]
                    capture_ms = float(prefetch_output.get("capture_ms", 0.0))
                    prepare_ms = float(prefetch_output.get("prepare_ms", 0.0))
                    prefetch_used = True
                    prefetch_output = None
                    current_state = _read_current_state(robot, args.expected_state_dim)
                else:
                    capture_start_t = time.perf_counter()
                    observation = robot.capture_observation()
                    capture_ms = (time.perf_counter() - capture_start_t) * 1000

                    prepare_start_t = time.perf_counter()
                    policy_observation, current_state = _build_policy_observation(
                        observation,
                        args.task,
                        args.image_size,
                        args.expected_state_dim,
                        args.obs_base_image_key,
                        args.obs_wrist_image_key,
                        args.obs_state_key,
                        args.policy_base_image_key,
                        args.policy_wrist_image_key,
                    )
                    prepare_ms = (time.perf_counter() - prepare_start_t) * 1000
                    chunk_result = _fetch_chunk_from_policy(
                        policy,
                        policy_observation,
                        action_key=args.policy_action_key,
                        expected_state_dim=args.expected_state_dim,
                        chunk_execute_steps=args.chunk_execute_steps,
                    )

                prev_infer_recv_t = latest_infer_recv_t
                latest_infer_recv_t = chunk_result["infer_recv_t"]
                infer_seq += 1
                pending_actions = chunk_result["pending_actions"]
                raw_chunk_len = chunk_result["raw_chunk_len"]
                executed_chunk_len = chunk_result["executed_chunk_len"]
                chunk_extract_ms = chunk_result["chunk_extract_ms"]
                policy_roundtrip_ms = chunk_result["policy_roundtrip_ms"]
                pending_action_index = 0
                latest_policy_timing = chunk_result["policy_timing"]
                latest_server_timing = chunk_result["server_timing"]
                infer_requested = True
                chunk_fetch_count += 1
                prefetch_started_chunk_idx = -1
                logging.info(
                    "Fetched action chunk from policy server prefetch_used=%s prefetch_wait_ms=%.2f "
                    "raw_chunk_len=%s executed_chunk_len=%s "
                    "capture_ms=%.2f prepare_ms=%.2f policy_roundtrip_ms=%.2f chunk_extract_ms=%.2f "
                    "server_infer_ms=%s policy_total_ms=%s "
                    "flow_prefix_image_embed_ms=%s flow_prompt_embed_ms=%s flow_prefix_concat_ms=%s "
                    "flow_prefix_embed_ms=%s "
                    "flow_prefix_prefill_ms=%s flow_loop_ms=%s flow_total_ms=%s",
                    prefetch_used,
                    prefetch_wait_ms,
                    raw_chunk_len,
                    executed_chunk_len,
                    capture_ms,
                    prepare_ms,
                    policy_roundtrip_ms,
                    chunk_extract_ms,
                    latest_server_timing.get("infer_ms"),
                    latest_policy_timing.get("total_ms"),
                    latest_policy_timing.get("flow_prefix_image_embed_ms"),
                    latest_policy_timing.get("flow_prompt_embed_ms"),
                    latest_policy_timing.get("flow_prefix_concat_ms"),
                    latest_policy_timing.get("flow_prefix_embed_ms"),
                    latest_policy_timing.get("flow_prefix_prefill_ms"),
                    latest_policy_timing.get("flow_loop_ms"),
                    latest_policy_timing.get("flow_total_ms"),
                )
            else:
                read_state_start_t = time.perf_counter()
                current_state = _read_current_state(robot, args.expected_state_dim)
                read_state_ms = (time.perf_counter() - read_state_start_t) * 1000

            action = np.array(pending_actions[pending_action_index], copy=True)
            chunk_step = pending_action_index + 1
            chunk_size = len(pending_actions)
            pending_action_index += 1

            if action.shape != current_state.shape:
                raise ValueError(
                    f"Policy returned {action.shape[0]} action dims, but Piper runtime is using {current_state.shape[0]}"
                )

            if not 0 <= args.hold_joint_index < action.shape[0]:
                raise ValueError(
                    f"hold_joint_index={args.hold_joint_index} is out of range for action dimension {action.shape[0]}"
                )
            action[args.hold_joint_index] = current_state[args.hold_joint_index]

            clamp_start_t = time.perf_counter()
            safe_action, was_clamped = _clamp_action_delta(
                action,
                current_state,
                max_abs_joint_delta=args.max_abs_joint_delta,
                max_abs_gripper_delta=args.max_abs_gripper_delta,
            )
            clamp_ms = (time.perf_counter() - clamp_start_t) * 1000

            filter_start_t = time.perf_counter()
            filtered_action, was_filtered = _low_pass_action(
                safe_action,
                previous_executed_action,
                alpha=args.action_filter_alpha,
            )
            filtered_action[args.hold_joint_index] = current_state[args.hold_joint_index]
            final_action, was_reclamped = _clamp_action_delta(
                filtered_action,
                current_state,
                max_abs_joint_delta=args.max_abs_joint_delta,
                max_abs_gripper_delta=args.max_abs_gripper_delta,
            )
            filter_ms = (time.perf_counter() - filter_start_t) * 1000
            if was_clamped:
                logging.warning(
                    "step=%s action clamp triggered | current_state=%s raw_action=%s safe_action=%s",
                    step_idx,
                    current_state.tolist(),
                    action.tolist(),
                    safe_action.tolist(),
                )
            if was_filtered and not np.allclose(filtered_action, safe_action):
                logging.debug(
                    "step=%s action filter applied | safe_action=%s filtered_action=%s",
                    step_idx,
                    safe_action.tolist(),
                    filtered_action.tolist(),
                )
            if was_reclamped and not np.allclose(final_action, filtered_action):
                logging.warning(
                    "step=%s filtered action clamp triggered | filtered_action=%s final_action=%s",
                    step_idx,
                    filtered_action.tolist(),
                    final_action.tolist(),
                )

            debounced_action, action_suppressed = _suppress_small_action_update(
                final_action,
                previous_executed_action,
                min_abs_joint_change=args.min_abs_joint_action_change,
                min_abs_gripper_change=args.min_abs_gripper_action_change,
            )
            if action_suppressed:
                executed_action = np.array(previous_executed_action, copy=True)
                logging.debug(
                    "step=%s action suppressed by deadband | previous_action=%s candidate_action=%s",
                    step_idx,
                    previous_executed_action.tolist() if previous_executed_action is not None else None,
                    final_action.tolist(),
                )
            else:
                action_send_start_t = time.perf_counter()
                executed_action = robot.send_action(torch.tensor(debounced_action, dtype=torch.float32))
                action_send_ms = (time.perf_counter() - action_send_start_t) * 1000
                executed_action = _to_numpy(executed_action).astype(np.float32)

            if executed_action.shape == debounced_action.shape:
                previous_executed_action = np.array(executed_action, copy=True)
            else:
                previous_executed_action = np.array(debounced_action, copy=True)

            if (
                not action_suppressed
                and executed_action.shape == debounced_action.shape
                and not np.allclose(executed_action, debounced_action)
            ):
                logging.warning(
                    "step=%s robot-level action clip triggered | final_action=%s executed_action=%s",
                    step_idx,
                    debounced_action.tolist(),
                    executed_action.tolist(),
                )

            elapsed_s = time.perf_counter() - loop_start_t
            sleep_s = 0.0
            if control_fps is not None and control_fps > 0:
                sleep_s = max(0.0, (1.0 / control_fps) - elapsed_s)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            loop_compute_ms = elapsed_s * 1000
            cycle_ms = (time.perf_counter() - loop_start_t) * 1000
            effective_hz = 1000.0 / cycle_ms if cycle_ms > 0 else float("inf")
            network_overhead_ms = None
            if infer_requested and latest_server_timing.get("total_ms") is not None:
                network_overhead_ms = policy_roundtrip_ms - float(latest_server_timing["total_ms"])
            elif infer_requested and latest_server_timing.get("infer_ms") is not None:
                network_overhead_ms = policy_roundtrip_ms - float(latest_server_timing["infer_ms"])

            if (
                args.prefetch_trigger_step > 0
                and prefetch_thread is None
                and pending_actions is not None
                and chunk_fetch_count > 0
                and prefetch_started_chunk_idx != chunk_fetch_count
                and chunk_step == args.prefetch_trigger_step
                and chunk_step < chunk_size
            ):
                pre_cap_start = time.perf_counter()
                pre_obs = robot.capture_observation()
                pre_capture_ms = (time.perf_counter() - pre_cap_start) * 1000

                pre_prep_start = time.perf_counter()
                pre_policy_obs, _ = _build_policy_observation(
                    pre_obs,
                    args.task,
                    args.image_size,
                    args.expected_state_dim,
                    args.obs_base_image_key,
                    args.obs_wrist_image_key,
                    args.obs_state_key,
                    args.policy_base_image_key,
                    args.policy_wrist_image_key,
                )
                pre_prepare_ms = (time.perf_counter() - pre_prep_start) * 1000

                prefetch_output = {
                    "capture_ms": pre_capture_ms,
                    "prepare_ms": pre_prepare_ms,
                }
                prefetch_thread = threading.Thread(
                    target=_prefetch_worker,
                    kwargs={
                        "policy": prefetch_policy,
                        "policy_observation": pre_policy_obs,
                        "action_key": args.policy_action_key,
                        "expected_state_dim": args.expected_state_dim,
                        "chunk_execute_steps": args.chunk_execute_steps,
                        "output": prefetch_output,
                    },
                    daemon=True,
                )
                prefetch_thread.start()
                prefetch_started_chunk_idx = chunk_fetch_count
                prefetch_started = True
                logging.info(
                    "Started next-chunk prefetch at chunk_step=%s/%s capture_ms=%.2f prepare_ms=%.2f",
                    chunk_step,
                    chunk_size,
                    pre_capture_ms,
                    pre_prepare_ms,
                )

            logging.info(
                "step=%s chunk_step=%s/%s infer_requested=%s prefetch_started=%s prefetch_used=%s prefetch_wait_ms=%.2f "
                "action_suppressed=%s raw_chunk_len=%s executed_chunk_len=%s "
                "cycle_ms=%.2f effective_hz=%.2f loop_compute_ms=%.2f sleep_ms=%.2f "
                "capture_ms=%.2f prepare_ms=%.2f read_state_ms=%.2f policy_roundtrip_ms=%.2f "
                "server_infer_ms=%s policy_ms=%s policy_total_ms=%s "
                "flow_prefix_image_embed_ms=%s flow_prompt_embed_ms=%s flow_prefix_concat_ms=%s "
                "flow_prefix_embed_ms=%s "
                "flow_prefix_prefill_ms=%s flow_loop_ms=%s flow_total_ms=%s "
                "network_overhead_ms=%s chunk_extract_ms=%.2f clamp_ms=%.2f filter_ms=%.2f action_send_ms=%.2f",
                step_idx,
                chunk_step,
                chunk_size,
                infer_requested,
                prefetch_started,
                prefetch_used,
                prefetch_wait_ms,
                action_suppressed,
                raw_chunk_len,
                executed_chunk_len,
                cycle_ms,
                effective_hz,
                loop_compute_ms,
                sleep_s * 1000,
                capture_ms,
                prepare_ms,
                read_state_ms,
                policy_roundtrip_ms,
                latest_server_timing.get("infer_ms"),
                latest_policy_timing.get("infer_ms"),
                latest_policy_timing.get("total_ms"),
                latest_policy_timing.get("flow_prefix_image_embed_ms"),
                latest_policy_timing.get("flow_prompt_embed_ms"),
                latest_policy_timing.get("flow_prefix_concat_ms"),
                latest_policy_timing.get("flow_prefix_embed_ms"),
                latest_policy_timing.get("flow_prefix_prefill_ms"),
                latest_policy_timing.get("flow_loop_ms"),
                latest_policy_timing.get("flow_total_ms"),
                network_overhead_ms,
                chunk_extract_ms,
                clamp_ms,
                filter_ms,
                action_send_ms,
            )

            inter_step_ms = None if prev_loop_start_t is None else (loop_start_t - prev_loop_start_t) * 1000
            infer_interval_ms = (
                None
                if latest_infer_recv_t is None or prev_infer_recv_t is None
                else (latest_infer_recv_t - prev_infer_recv_t) * 1000
            )
            action_age_ms = (
                None if latest_infer_recv_t is None else max(0.0, (loop_start_t - latest_infer_recv_t) * 1000)
            )

            if args.trace_csv_path is not None:
                if trace_writer is None:
                    trace_state_dim = int(current_state.shape[0])
                    args.trace_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_file = args.trace_csv_path.open("w", newline="")
                    trace_writer = csv.DictWriter(trace_file, fieldnames=_make_trace_fieldnames(trace_state_dim))
                    trace_writer.writeheader()

                row = {
                    "step_idx": step_idx,
                    "wall_time_s": time.time(),
                    "monotonic_s": loop_start_t,
                    "t_rel_s": loop_start_t - start_t,
                    "inter_step_ms": inter_step_ms,
                    "infer_seq": infer_seq,
                    "infer_requested": infer_requested,
                    "prefetch_started": prefetch_started,
                    "prefetch_used": prefetch_used,
                    "prefetch_wait_ms": prefetch_wait_ms,
                    "infer_interval_ms": infer_interval_ms,
                    "action_age_ms": action_age_ms,
                    "chunk_step": chunk_step,
                    "chunk_size": chunk_size,
                    "action_suppressed": action_suppressed,
                    "was_clamped": was_clamped,
                    "was_filtered": was_filtered,
                    "was_reclamped": was_reclamped,
                    "capture_ms": capture_ms,
                    "prepare_ms": prepare_ms,
                    "read_state_ms": read_state_ms,
                    "policy_roundtrip_ms": policy_roundtrip_ms,
                    "chunk_extract_ms": chunk_extract_ms,
                    "clamp_ms": clamp_ms,
                    "filter_ms": filter_ms,
                    "action_send_ms": action_send_ms,
                    "loop_compute_ms": loop_compute_ms,
                    "cycle_ms": cycle_ms,
                    "sleep_ms": sleep_s * 1000,
                    "effective_hz": effective_hz,
                    "network_overhead_ms": network_overhead_ms,
                    "server_infer_ms": latest_server_timing.get("infer_ms"),
                    "server_total_ms": latest_server_timing.get("total_ms"),
                    "policy_total_ms": latest_policy_timing.get("total_ms"),
                }
                _write_trace_vector(row, "current_state", current_state, trace_state_dim)
                _write_trace_vector(row, "raw_action", action, trace_state_dim)
                _write_trace_vector(row, "final_action", debounced_action, trace_state_dim)
                _write_trace_vector(row, "executed_action", executed_action, trace_state_dim)
                trace_writer.writerow(row)
                if step_idx % 50 == 0:
                    trace_file.flush()

            step_idx += 1
            prev_loop_start_t = loop_start_t
    finally:
        if prefetch_thread is not None:
            prefetch_thread.join(timeout=0.2)
        robot.disconnect()
        if trace_file is not None:
            trace_file.flush()
            trace_file.close()


if __name__ == "__main__":
    main()

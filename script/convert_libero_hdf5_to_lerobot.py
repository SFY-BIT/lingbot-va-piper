#!/usr/bin/env python3
"""
Convert LIBERO raw HDF5 demos into a LeRobot-style dataset skeleton that is
compatible with LingBot-VA's dataset loader.

This script handles:
1. reading raw LIBERO *.hdf5 demos,
2. writing episode-level parquet files,
3. exporting resized MP4 videos for the two LIBERO cameras,
4. creating metadata files (info.json / tasks.jsonl / episodes.jsonl),
5. enforcing the Wan latent extraction frame-count rule: num_frames % 4 == 1.

It does NOT extract Wan latents. That should be done afterwards against the
generated videos/ directory.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image


SOURCE_CAMERAS = {
    "observation.images.image": "agentview_rgb",
    "observation.images.wrist_image": "eye_in_hand_rgb",
}

ACTION_NAMES = [
    "x",
    "y",
    "z",
    "axis_angle1",
    "axis_angle2",
    "axis_angle3",
    "gripper",
]

STATE_NAMES = [
    "x",
    "y",
    "z",
    "axis_angle1",
    "axis_angle2",
    "axis_angle3",
    "gripper",
    "gripper",
]


@dataclass
class EpisodeRecord:
    episode_index: int
    task_index: int
    task: str
    length: int
    chunk_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 to LeRobot format")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing LIBERO raw *.hdf5 files, e.g. libero_raw/libero_spatial",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output LeRobot-style dataset directory",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=20,
        help="Target FPS stored in metadata and output videos",
    )
    parser.add_argument(
        "--source-fps",
        type=int,
        default=20,
        help="Assumed FPS of raw LIBERO demos",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Output video height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Output video width",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Episode chunk size used for data/ and videos/ layout",
    )
    parser.add_argument(
        "--frame-policy",
        choices=["trim", "pad", "none"],
        default="trim",
        help="How to enforce video_num_frames %% 4 == 1 before latent extraction",
    )
    parser.add_argument(
        "--empty-emb-shape",
        type=int,
        nargs=2,
        default=(512, 4096),
        metavar=("L", "D"),
        help="Shape of empty_emb.pt to generate",
    )
    return parser.parse_args()


def numeric_demo_sort_key(name: str) -> Tuple[int, str]:
    try:
        return int(name.split("_")[-1]), name
    except Exception:
        return 10**9, name


def task_from_filename(file_path: Path) -> str:
    stem = file_path.stem
    if stem.endswith("_demo"):
        stem = stem[:-5]
    return stem.replace("_", " ")


def ensure_frame_rule(frame_ids: np.ndarray, policy: str) -> np.ndarray:
    if policy == "none":
        return frame_ids

    mod = len(frame_ids) % 4
    if mod == 1:
        return frame_ids

    if policy == "trim":
        target = len(frame_ids) - ((mod - 1) % 4)
        target = max(target, 1)
        return frame_ids[:target]

    # pad
    needed = (1 - mod) % 4
    pad = np.repeat(frame_ids[-1:], needed)
    return np.concatenate([frame_ids, pad], axis=0)


def sample_frame_ids(total_frames: int, source_fps: int, target_fps: int, policy: str) -> np.ndarray:
    if target_fps <= 0 or source_fps <= 0:
        raise ValueError("FPS must be positive")
    if target_fps >= source_fps:
        frame_ids = np.arange(total_frames, dtype=np.int64)
    else:
        step = source_fps / float(target_fps)
        count = max(1, int(math.ceil(total_frames / step)))
        frame_ids = np.unique(np.clip(np.round(np.arange(count) * step).astype(np.int64), 0, total_frames - 1))
    return ensure_frame_rule(frame_ids, policy)


def resize_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    out = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize((width, height), Image.BILINEAR)
        out.append(np.asarray(img, dtype=np.uint8))
    return np.stack(out, axis=0)


def compute_video_stats(frames: np.ndarray) -> Dict[str, List]:
    arr = frames.astype(np.float32) / 255.0
    arr = np.transpose(arr, (3, 0, 1, 2))
    mins = arr.min(axis=(1, 2, 3))
    maxs = arr.max(axis=(1, 2, 3))
    means = arr.mean(axis=(1, 2, 3))
    stds = arr.std(axis=(1, 2, 3))
    count = [int(arr.shape[1])]
    return {
        "min": [[[float(v)]] for v in mins],
        "max": [[[float(v)]] for v in maxs],
        "mean": [[[float(v)]] for v in means],
        "std": [[[float(v)]] for v in stds],
        "count": count,
    }


def compute_array_stats(arr: np.ndarray) -> Dict[str, List]:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }


def write_video(video_path: Path, frames: np.ndarray, fps: int) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    # imageio may use different backends across environments; PyAV does not
    # accept the same quality arguments as ffmpeg.
    try:
        imageio.mimwrite(video_path, frames, fps=fps, codec="libx264", quality=8)
    except TypeError:
        imageio.mimwrite(video_path, frames, fps=fps, codec="libx264")


def build_episode_dataframe(
    observation_state: np.ndarray,
    actions: np.ndarray,
    frame_ids: np.ndarray,
    episode_index: int,
    task_index: int,
    global_index_start: int,
    target_fps: int,
) -> pd.DataFrame:
    timestamps = frame_ids.astype(np.float32) / float(target_fps)
    frame_index = np.arange(len(frame_ids), dtype=np.int64)
    global_index = np.arange(global_index_start, global_index_start + len(frame_ids), dtype=np.int64)
    data = {
        "observation.state": list(observation_state.astype(np.float32)),
        "action": list(actions.astype(np.float32)),
        "timestamp": timestamps,
        "frame_index": frame_index,
        "episode_index": np.full(len(frame_ids), episode_index, dtype=np.int64),
        "index": global_index,
        "task_index": np.full(len(frame_ids), task_index, dtype=np.int64),
    }
    return pd.DataFrame(data)


def build_info(total_episodes: int, total_frames: int, total_tasks: int, total_chunks: int, chunk_size: int, fps: int, height: int, width: int) -> Dict:
    return {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(SOURCE_CAMERAS),
        "total_chunks": total_chunks,
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.image": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.images.wrist_image": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": {"motors": STATE_NAMES},
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": ACTION_NAMES},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def prepare_dirs(root: Path) -> None:
    for subdir in ["meta", "data", "videos", "latents"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    prepare_dirs(args.output_dir)

    hdf5_files = sorted(args.input_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {args.input_dir}")

    task_to_index: Dict[str, int] = {}
    episode_records: List[EpisodeRecord] = []
    episode_stats: List[Dict] = []
    global_index = 0
    episode_index = 0

    for hdf5_path in hdf5_files:
        task = task_from_filename(hdf5_path)
        task_index = task_to_index.setdefault(task, len(task_to_index))
        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(f["data"].keys(), key=numeric_demo_sort_key)
            for demo_name in demos:
                demo = f["data"][demo_name]
                raw_actions = np.asarray(demo["actions"], dtype=np.float32)
                raw_agent = np.asarray(demo["obs"]["agentview_rgb"], dtype=np.uint8)
                raw_wrist = np.asarray(demo["obs"]["eye_in_hand_rgb"], dtype=np.uint8)
                ee_pos = np.asarray(demo["obs"]["ee_pos"], dtype=np.float32)
                ee_ori = np.asarray(demo["obs"]["ee_ori"], dtype=np.float32)
                gripper = np.asarray(demo["obs"]["gripper_states"], dtype=np.float32)

                frame_ids = sample_frame_ids(
                    total_frames=raw_actions.shape[0],
                    source_fps=args.source_fps,
                    target_fps=args.target_fps,
                    policy=args.frame_policy,
                )

                actions = raw_actions[frame_ids]
                agent_frames = resize_frames(raw_agent[np.clip(frame_ids, 0, len(raw_agent) - 1)], args.height, args.width)
                wrist_frames = resize_frames(raw_wrist[np.clip(frame_ids, 0, len(raw_wrist) - 1)], args.height, args.width)
                observation_state = np.concatenate([ee_pos[frame_ids], ee_ori[frame_ids], gripper[frame_ids]], axis=1).astype(np.float32)

                chunk_index = episode_index // args.chunk_size
                chunk_dir = f"chunk-{chunk_index:03d}"
                parquet_path = args.output_dir / "data" / chunk_dir / f"episode_{episode_index:06d}.parquet"
                parquet_path.parent.mkdir(parents=True, exist_ok=True)

                df = build_episode_dataframe(
                    observation_state=observation_state,
                    actions=actions,
                    frame_ids=frame_ids,
                    episode_index=episode_index,
                    task_index=task_index,
                    global_index_start=global_index,
                    target_fps=args.target_fps,
                )
                df.to_parquet(parquet_path, index=False)

                for video_key, source_name in SOURCE_CAMERAS.items():
                    frames = agent_frames if source_name == "agentview_rgb" else wrist_frames
                    video_path = args.output_dir / "videos" / chunk_dir / video_key / f"episode_{episode_index:06d}.mp4"
                    write_video(video_path, frames, args.target_fps)

                length = int(len(frame_ids))
                episode_records.append(
                    EpisodeRecord(
                        episode_index=episode_index,
                        task_index=task_index,
                        task=task,
                        length=length,
                        chunk_index=chunk_index,
                    )
                )

                episode_stats.append(
                    {
                        "episode_index": episode_index,
                        "stats": {
                            "observation.images.wrist_image": compute_video_stats(wrist_frames),
                            "observation.images.image": compute_video_stats(agent_frames),
                            "observation.state": compute_array_stats(observation_state),
                            "action": compute_array_stats(actions),
                            "timestamp": compute_array_stats(df["timestamp"].to_numpy(dtype=np.float32)),
                            "frame_index": compute_array_stats(df["frame_index"].to_numpy(dtype=np.int64)),
                            "episode_index": compute_array_stats(df["episode_index"].to_numpy(dtype=np.int64)),
                            "index": compute_array_stats(df["index"].to_numpy(dtype=np.int64)),
                            "task_index": compute_array_stats(df["task_index"].to_numpy(dtype=np.int64)),
                        },
                    }
                )

                global_index += length
                episode_index += 1

    meta_dir = args.output_dir / "meta"
    info = build_info(
        total_episodes=len(episode_records),
        total_frames=global_index,
        total_tasks=len(task_to_index),
        total_chunks=max((r.chunk_index for r in episode_records), default=-1) + 1,
        chunk_size=args.chunk_size,
        fps=args.target_fps,
        height=args.height,
        width=args.width,
    )
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    task_lines = [
        {"task_index": idx, "task": task}
        for task, idx in sorted(task_to_index.items(), key=lambda kv: kv[1])
    ]
    for file_name in ["tasks.jsonl"]:
        with open(meta_dir / file_name, "w") as f:
            for line in task_lines:
                f.write(json.dumps(line) + "\n")

    episodes_lines = []
    for rec in episode_records:
        episodes_lines.append(
            {
                "episode_index": rec.episode_index,
                "tasks": [rec.task],
                "length": rec.length,
                "action_config": [
                    {
                        "start_frame": 0,
                        "end_frame": rec.length,
                        "action_text": rec.task,
                    }
                ],
            }
        )

    for file_name in ["episodes.jsonl", "episodes_with_action_config.jsonl"]:
        with open(meta_dir / file_name, "w") as f:
            for line in episodes_lines:
                f.write(json.dumps(line) + "\n")

    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for line in episode_stats:
            f.write(json.dumps(line) + "\n")

    empty_emb = torch.zeros(args.empty_emb_shape, dtype=torch.bfloat16)
    torch.save(empty_emb, args.output_dir / "empty_emb.pt")

    print(f"Wrote LeRobot-style dataset to: {args.output_dir}")
    print(f"Total episodes: {len(episode_records)}")
    print(f"Total tasks: {len(task_to_index)}")
    print(f"Total frames after frame-policy '{args.frame_policy}': {global_index}")
    print("Next step: extract Wan2.2 latents into output_dir/latents/ matching the videos layout.")


if __name__ == "__main__":
    main()

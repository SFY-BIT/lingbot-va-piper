#!/usr/bin/env python3
"""
Run the LIBERO data preparation pipeline in a fixed sequential order.

Default order:
1. extract latents for spatial
2. convert goal raw hdf5 -> LeRobot
3. convert object raw hdf5 -> LeRobot
4. extract latents for goal
5. extract latents for object
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIBERO conversion + latent extraction sequentially")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/hdd/sfy/datasets/libero_raw"),
        help="Root containing libero_goal / libero_object / libero_spatial raw data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/hdd/sfy/datasets"),
        help="Root for LeRobot-format outputs",
    )
    parser.add_argument(
        "--spatial-dataset-dir",
        type=Path,
        default=Path("/mnt/hdd/sfy/datasets/libero_spatial_lerobot"),
        help="Existing spatial LeRobot dataset directory used for latent extraction",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/mnt/hdd/sfy/models/lingbot-va-base"),
        help="Model root containing vae / text_encoder / tokenizer",
    )
    parser.add_argument("--source-fps", type=int, default=20)
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument(
        "--frame-policy",
        choices=["trim", "pad", "none"],
        default="trim",
    )
    parser.add_argument(
        "--latent-device",
        default="cuda",
        help="Device for latent extraction",
    )
    parser.add_argument(
        "--latent-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="float16",
        help="Dtype for latent extraction",
    )
    parser.add_argument(
        "--latent-cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for latent extraction, e.g. 1",
    )
    parser.add_argument(
        "--latent-overwrite",
        action="store_true",
        help="Overwrite existing latent files",
    )
    parser.add_argument(
        "--alloc-conf",
        default="expandable_segments:True",
        help="Value for PYTORCH_ALLOC_CONF during latent extraction",
    )
    return parser.parse_args()


def run_step(title: str, cmd: List[str], env: dict | None = None) -> None:
    print(f"\n=== {title} ===")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"=== Finished: {title} ===")


def build_convert_cmd(script_path: Path, input_dir: Path, output_dir: Path, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(script_path),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--source-fps",
        str(args.source_fps),
        "--target-fps",
        str(args.target_fps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--frame-policy",
        args.frame_policy,
        "--chunk-size",
        str(args.chunk_size),
    ]


def build_latent_cmd(script_path: Path, dataset_dir: Path, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset-dir",
        str(dataset_dir),
        "--model-dir",
        str(args.model_dir),
        "--fps",
        str(args.target_fps),
        "--ori-fps",
        str(args.source_fps),
        "--device",
        args.latent_device,
        "--dtype",
        args.latent_dtype,
        "--vae-frame-chunk",
        "0",
    ]
    if args.latent_overwrite:
        cmd.append("--overwrite")
    return cmd


def build_latent_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    if args.latent_cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.latent_cuda_visible_devices
    if args.alloc_conf:
        env["PYTORCH_ALLOC_CONF"] = args.alloc_conf
    return env


def main() -> None:
    args = parse_args()
    convert_script = Path(__file__).with_name("convert_libero_hdf5_to_lerobot.py")
    latent_script = Path(__file__).with_name("extract_wan_latents_from_lerobot.py")

    goal_raw = args.raw_root / "libero_goal"
    object_raw = args.raw_root / "libero_object"
    goal_lerobot = args.output_root / "libero_goal_lerobot"
    object_lerobot = args.output_root / "libero_object_lerobot"
    spatial_lerobot = args.spatial_dataset_dir

    if not spatial_lerobot.exists():
        raise FileNotFoundError(f"Missing spatial dataset dir: {spatial_lerobot}")
    if not goal_raw.exists():
        raise FileNotFoundError(f"Missing raw goal dir: {goal_raw}")
    if not object_raw.exists():
        raise FileNotFoundError(f"Missing raw object dir: {object_raw}")

    latent_env = build_latent_env(args)

    run_step(
        "Extract spatial latents",
        build_latent_cmd(latent_script, spatial_lerobot, args),
        env=latent_env,
    )

    run_step(
        "Convert goal raw to LeRobot",
        build_convert_cmd(convert_script, goal_raw, goal_lerobot, args),
    )

    run_step(
        "Convert object raw to LeRobot",
        build_convert_cmd(convert_script, object_raw, object_lerobot, args),
    )

    run_step(
        "Extract goal latents",
        build_latent_cmd(latent_script, goal_lerobot, args),
        env=latent_env,
    )

    run_step(
        "Extract object latents",
        build_latent_cmd(latent_script, object_lerobot, args),
        env=latent_env,
    )


if __name__ == "__main__":
    main()

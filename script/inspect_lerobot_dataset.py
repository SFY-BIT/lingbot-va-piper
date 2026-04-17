#!/usr/bin/env python3
"""
Quick inspection utility for a LeRobot-style dataset prepared for LingBot-VA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a LeRobot-style dataset")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--check-latents", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_dir
    meta = root / "meta"

    print("dataset_dir:", root)
    print("has info.json:", (meta / "info.json").exists())
    print("has episodes.jsonl:", (meta / "episodes.jsonl").exists())
    print("has empty_emb.pt:", (root / "empty_emb.pt").exists())

    if (meta / "info.json").exists():
        info = json.loads((meta / "info.json").read_text())
        print("fps:", info.get("fps"))
        print("features:", sorted(info.get("features", {}).keys()))

    if (meta / "episodes.jsonl").exists():
        lengths = []
        episode_count = 0
        action_cfg_episode_count = 0
        action_cfg_segment_count = 0
        with open(meta / "episodes.jsonl") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                episode_count += 1
                lengths.append(obj["length"])
                segs = obj.get("action_config") or []
                if segs:
                    action_cfg_episode_count += 1
                    action_cfg_segment_count += len(segs)
                if i + 1 >= args.limit:
                    break
        print(f"episodes (sampled): {episode_count}")
        print(
            "episodes with action_config (sampled):",
            f"{action_cfg_episode_count}/{episode_count}",
        )
        print("action_config segments (sampled):", action_cfg_segment_count)
        print("sample lengths:", lengths)
        print("length % 4:", [x % 4 for x in lengths])

    if args.check_latents:
        latent_files = sorted((root / "latents").glob("chunk-*/*/*.pth"))
        print("latent files found:", len(latent_files))
        for p in latent_files[: args.limit]:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            print(
                p.name,
                "video_num_frames=", obj.get("video_num_frames"),
                "mod4=", obj.get("video_num_frames", 0) % 4 if "video_num_frames" in obj else None,
                "latent_num_frames=", obj.get("latent_num_frames"),
            )


if __name__ == "__main__":
    main()

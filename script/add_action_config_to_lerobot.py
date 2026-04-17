#!/usr/bin/env python3
"""Add LingBot-VA action_config segments to a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add action_config to meta/episodes.jsonl for LingBot-VA latent extraction."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="LeRobot dataset root containing meta/episodes.jsonl",
    )
    parser.add_argument(
        "--action-text",
        default=None,
        help="Use this text for every segment. Defaults to the episode's first task.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing action_config entries if present.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create meta/episodes.jsonl.bak before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes_path = args.dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing {episodes_path}")

    episodes = []
    changed = 0
    with open(episodes_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            episode = json.loads(line)
            if "action_config" in episode and not args.overwrite:
                episodes.append(episode)
                continue

            tasks = episode.get("tasks") or []
            action_text = args.action_text if args.action_text is not None else (tasks[0] if tasks else "")
            length = int(episode["length"])
            episode["action_config"] = [
                {
                    "start_frame": 0,
                    "end_frame": length,
                    "action_text": action_text,
                }
            ]
            episodes.append(episode)
            changed += 1

    if changed == 0:
        print(f"No changes needed: {episodes_path}")
        return

    if not args.no_backup:
        backup_path = episodes_path.with_suffix(episodes_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(episodes_path, backup_path)
            print(f"Backed up original episodes file to: {backup_path}")

    with open(episodes_path, "w") as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    print(f"Updated {changed} episodes in: {episodes_path}")


if __name__ == "__main__":
    main()

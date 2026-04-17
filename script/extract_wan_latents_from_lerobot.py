#!/usr/bin/env python3
"""
Extract Wan2.2 latents from a LeRobot-style dataset for LingBot-VA training.

The script expects a dataset layout like:

dataset/
  meta/episodes.jsonl
  videos/chunk-000/<camera_key>/episode_000000.mp4

It writes:

dataset/
  latents/chunk-000/<camera_key>/episode_000000_0_109.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wan_va.modules.utils import (
    load_text_encoder,
    load_tokenizer,
    load_vae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Wan latents from LeRobot videos")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Converted LeRobot-style dataset root",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/mnt/hdd/sfy/models/lingbot-va-base"),
        help="Wan/LingBot model root containing vae, text_encoder, tokenizer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output latents root, defaults to <dataset-dir>/latents",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Target fps for frame sampling and latent metadata",
    )
    parser.add_argument(
        "--ori-fps",
        type=int,
        default=None,
        help="Original episode fps. Defaults to meta/info.json fps when available.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Video frame height fed to the VAE",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Video frame width fed to the VAE",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Text embedding sequence length",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model inference dtype",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing latent files",
    )
    parser.add_argument(
        "--vae-frame-chunk",
        type=int,
        default=17,
        help="Reserved for compatibility; current encoder processes each sampled segment at once",
    )
    parser.add_argument(
        "--frame-policy",
        choices=["trim", "none"],
        default="trim",
        help="How to handle sampled frame counts that do not satisfy num_frames %% 4 == 1",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Process at most this many episodes, useful for smoke tests",
    )
    parser.add_argument(
        "--empty-emb-path",
        type=Path,
        default=None,
        help="Where to write an all-zero empty text embedding, defaults to <dataset-dir>/empty_emb.pt",
    )
    parser.add_argument(
        "--auto-action-config",
        action="store_true",
        help=(
            "If an episode has no action_config, auto-create one segment "
            "[0, length) using the first task text."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def find_video_path(dataset_dir: Path, camera_key: str, episode_index: int) -> Path:
    pattern = f"episode_{episode_index:06d}.mp4"
    matches = sorted((dataset_dir / "videos").glob(f"chunk-*/{camera_key}/{pattern}"))
    if not matches:
        raise FileNotFoundError(f"Missing video for episode {episode_index}: {camera_key}/{pattern}")
    return matches[0]


def encode_prompt(
    text_encoder,
    tokenizer,
    text: str,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> torch.Tensor:
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    seq_len = int(attention_mask.gt(0).sum(dim=1)[0].item())
    with torch.inference_mode():
        prompt_embeds = text_encoder(text_input_ids, attention_mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype).clone()
    prompt_embeds = prompt_embeds[0]
    if seq_len < max_sequence_length:
        prompt_embeds[seq_len:] = 0
    return prompt_embeds.cpu().to(torch.bfloat16)


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    image = Image.fromarray(frame)
    return np.asarray(image.resize((width, height), Image.BICUBIC))


def sample_frame_ids(
    start_frame: int,
    end_frame: int,
    ori_fps: int,
    target_fps: int,
    frame_policy: str,
) -> List[int]:
    if target_fps <= 0:
        raise ValueError("--fps must be positive")
    if ori_fps <= 0:
        raise ValueError("--ori-fps must be positive")
    stride = max(1, int(round(ori_fps / target_fps)))
    frame_ids = list(range(start_frame, end_frame, stride))
    if frame_policy == "trim":
        remainder = len(frame_ids) % 4
        if remainder != 1:
            trim_count = (remainder - 1) % 4
            if trim_count:
                frame_ids = frame_ids[:-trim_count]
    return frame_ids


def get_episode_segments(episode: dict, auto_action_config: bool) -> List[dict]:
    action_config = episode.get("action_config") or []
    if action_config:
        return action_config

    if not auto_action_config:
        episode_index = int(episode.get("episode_index", -1))
        raise ValueError(
            "Missing action_config in episodes.jsonl "
            f"(episode_index={episode_index}). "
            "Run script/add_action_config_to_lerobot.py first, or pass "
            "--auto-action-config to create one full-length segment per episode."
        )

    tasks = episode.get("tasks", [])
    action_text = tasks[0] if tasks else ""
    length = int(episode["length"])
    return [{"start_frame": 0, "end_frame": length, "action_text": action_text}]


def validate_segment(segment: dict, episode_index: int, episode_length: int) -> None:
    start_frame = int(segment["start_frame"])
    end_frame = int(segment["end_frame"])
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError(
            f"Invalid segment in episode {episode_index}: "
            f"start={start_frame}, end={end_frame}"
        )
    if end_frame > episode_length:
        raise ValueError(
            f"Segment end exceeds episode length in episode {episode_index}: "
            f"end={end_frame}, length={episode_length}"
        )


def normalize_latents(mu: torch.Tensor, vae) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=mu.dtype)
    latents_std = torch.tensor(vae.config.latents_std, device=mu.device, dtype=mu.dtype)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    return ((mu.float() - latents_mean) * (1.0 / latents_std)).to(mu.dtype)


def encode_video_segment(
    vae,
    frames,
    device: torch.device,
    dtype: torch.dtype,
    frame_chunk: int,
) -> Dict[str, object]:
    if not isinstance(frames, np.ndarray):
        frames = np.stack(frames, axis=0)
    with torch.inference_mode():
        video = torch.from_numpy(frames).float().permute(3, 0, 1, 2).unsqueeze(0)
        video = video / 255.0 * 2.0 - 1.0
        enc_out = vae._encode(video.to(device=device, dtype=dtype))
        mu, _ = torch.chunk(enc_out, 2, dim=1)
        mu_norm = normalize_latents(mu, vae)
    latent = mu_norm[0].permute(1, 2, 3, 0).reshape(-1, mu_norm.shape[1]).cpu().to(torch.bfloat16)
    return {
        "latent": latent,
        "latent_num_frames": int(mu_norm.shape[2]),
        "latent_height": int(mu_norm.shape[3]),
        "latent_width": int(mu_norm.shape[4]),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = (args.output_dir or (dataset_dir / "latents")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = dataset_dir / "meta" / "info.json"
    info = read_json(info_path) if info_path.exists() else {}
    ori_fps = int(args.ori_fps or info.get("fps", args.fps))

    device = torch.device(args.device)
    dtype = torch_dtype_from_name(args.dtype)

    vae = load_vae(args.model_dir / "vae", dtype, device)
    text_device = torch.device("cpu")
    text_encoder = load_text_encoder(args.model_dir / "text_encoder", torch.float32, text_device)
    tokenizer = load_tokenizer(args.model_dir / "tokenizer")
    empty_emb = encode_prompt(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text="",
        device=text_device,
        dtype=torch.float32,
        max_sequence_length=args.max_seq_len,
    )
    empty_emb.zero_()
    empty_emb_path = (args.empty_emb_path or (dataset_dir / "empty_emb.pt")).resolve()
    torch.save(empty_emb, empty_emb_path)
    print(f"Wrote empty text embedding: {empty_emb_path}")

    episodes = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    camera_dirs = sorted((dataset_dir / "videos").glob("chunk-*/*"))
    camera_keys = sorted({p.name for p in camera_dirs if p.is_dir()})
    if not camera_keys:
        raise FileNotFoundError(f"No camera directories found under {dataset_dir / 'videos'}")

    episode_segments = {}
    total_segments = 0
    for ep in episodes:
        episode_index = int(ep["episode_index"])
        segs = get_episode_segments(ep, args.auto_action_config)
        episode_segments[episode_index] = segs
        total_segments += len(segs)
    total_jobs = total_segments * len(camera_keys)
    if total_jobs == 0:
        raise ValueError(
            "No segments found to process. Check episodes.jsonl/action_config."
        )
    text_emb_cache: Dict[str, torch.Tensor] = {}
    written_files = 0
    skipped_files = 0

    progress = tqdm(total=total_jobs, desc="Latents", unit="file", mininterval=5.0)
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        tasks = episode.get("tasks", [])
        default_text = tasks[0] if tasks else ""

        video_cache = {}
        for camera_key in camera_keys:
            video_path = find_video_path(dataset_dir, camera_key, episode_index)
            video_cache[camera_key] = imageio.mimread(video_path, memtest=False)

        episode_length = int(episode["length"])
        for seg in episode_segments[episode_index]:
            validate_segment(seg, episode_index, episode_length)
            start_frame = int(seg["start_frame"])
            end_frame = int(seg["end_frame"])
            action_text = seg.get("action_text", default_text)
            if action_text not in text_emb_cache:
                text_emb_cache[action_text] = encode_prompt(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    text=action_text,
                    device=text_device,
                    dtype=torch.float32,
                    max_sequence_length=args.max_seq_len,
                )
            text_emb = text_emb_cache[action_text]

            for camera_key in camera_keys:
                frame_ids = sample_frame_ids(
                    start_frame,
                    end_frame,
                    ori_fps,
                    args.fps,
                    args.frame_policy,
                )
                frames = [
                    resize_frame(video_cache[camera_key][frame_id], args.width, args.height)
                    for frame_id in frame_ids
                    if frame_id < len(video_cache[camera_key])
                ]
                if len(frames) == 0:
                    raise ValueError(
                        f"Empty segment for episode={episode_index}, camera={camera_key}, "
                        f"start={start_frame}, end={end_frame}"
                    )

                video_path = find_video_path(dataset_dir, camera_key, episode_index)
                chunk_dir = video_path.parent.parent.name
                target_dir = output_dir / chunk_dir / camera_key
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / (
                    f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                )
                if target_file.exists() and not args.overwrite:
                    skipped_files += 1
                    progress.update(1)
                    continue

                encoded = encode_video_segment(
                    vae=vae,
                    frames=frames,
                    device=device,
                    dtype=dtype,
                    frame_chunk=args.vae_frame_chunk,
                )

                payload = {
                    **encoded,
                    "video_num_frames": len(frames),
                    "video_height": int(frames[0].shape[0]),
                    "video_width": int(frames[0].shape[1]),
                    "text_emb": text_emb.clone(),
                    "text": action_text,
                    "frame_ids": frame_ids,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": args.fps,
                    "ori_fps": ori_fps,
                }
                torch.save(payload, target_file)
                written_files += 1
                progress.update(1)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    del text_encoder
    del tokenizer
    progress.close()
    print(f"Finished. written={written_files} skipped={skipped_files}")


if __name__ == "__main__":
    main()

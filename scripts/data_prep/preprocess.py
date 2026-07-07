"""Preprocess ETH/UCY trajectory data into ms-swift SFT format.

ETH/UCY format: tab-separated, each row = [frame_id, pedestrian_id, x, y]
Frame rate: 2.5 fps (one frame every 0.4 seconds)

Output format: JSON lines compatible with ms-swift SFT training.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from trajectory_norm import normalize_obs_pred, MODES  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "eth_ucy"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

SYSTEM_PROMPT = (
    "你是一个专业的行人轨迹预测专家。根据给定的行人历史轨迹和周围环境信息，"
    "预测该行人未来的运动轨迹。你需要先分析运动趋势，再给出预测坐标。"
)

OBS_LENGTH = 8       # 8 frames of observation (3.2 seconds)
PRED_LENGTH = 12     # 12 frames of prediction (4.8 seconds)
FRAME_INTERVAL = 0.4 # seconds per frame


def parse_eth_ucy(filepath: Path):
    """Parse ETH/UCY txt file into per-pedestrian trajectories.

    Returns: dict[ped_id] -> list of (frame, x, y)
    """
    trajectories = defaultdict(list)
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            frame_id = int(float(parts[0]))
            ped_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            trajectories[ped_id].append((frame_id, x, y))

    for ped_id in trajectories:
        trajectories[ped_id].sort(key=lambda t: t[0])

    return trajectories


def extract_samples(trajectories: dict, scene_name: str,
                    obs_len: int = OBS_LENGTH,
                    pred_len: int = PRED_LENGTH,
                    normalize: str = "none"):
    """Extract observation/prediction pairs from trajectories.

    For each pedestrian, slide a window of obs_len + pred_len frames.
    `normalize` applies an agent-centric transform (see trajectory_norm).
    """
    samples = []
    total_needed = obs_len + pred_len

    for ped_id, traj in trajectories.items():
        if len(traj) < total_needed:
            continue

        frames = [t[0] for t in traj]
        coords = np.array([(t[1], t[2]) for t in traj])

        # Detect this pedestrian's frame sampling step. ETH/UCY annotates every
        # 10 frames (0.4s @ 25fps); the synthetic generator uses step 1. A
        # hardcoded step of 1 here silently drops every real-data window.
        diffs = [frames[j + 1] - frames[j] for j in range(len(frames) - 1)]
        step = min((d for d in diffs if d > 0), default=1)

        for i in range(len(traj) - total_needed + 1):
            window = frames[i:i + total_needed]
            # require uniformly spaced, gap-free consecutive frames
            if any(window[k + 1] - window[k] != step
                   for k in range(total_needed - 1)):
                continue

            obs_coords = coords[i:i + obs_len]
            pred_coords = coords[i + obs_len:i + total_needed]

            # Apply the agent-centric transform (identity when normalize="none").
            # Speed/direction below are computed from the normalized frame so the
            # text prompt is self-consistent with the coordinates shown.
            obs_coords, pred_coords, _ = normalize_obs_pred(
                obs_coords, pred_coords, normalize)

            # Compute velocity and direction
            vel = np.diff(obs_coords, axis=0)
            avg_speed = np.mean(np.linalg.norm(vel, axis=1)) / FRAME_INTERVAL
            direction = np.arctan2(
                obs_coords[-1, 1] - obs_coords[0, 1],
                obs_coords[-1, 0] - obs_coords[0, 0]
            )
            dir_label = _direction_label(direction)

            sample = {
                "scene": scene_name,
                "ped_id": ped_id,
                "obs": obs_coords.tolist(),
                "pred": pred_coords.tolist(),
                "avg_speed": round(avg_speed, 2),
                "direction": dir_label,
            }
            samples.append(sample)

    return samples


def _direction_label(angle_rad: float) -> str:
    """Convert angle in radians to a Chinese direction label."""
    deg = np.degrees(angle_rad) % 360
    if 337.5 <= deg or deg < 22.5:
        return "东"
    elif 22.5 <= deg < 67.5:
        return "东北"
    elif 67.5 <= deg < 112.5:
        return "北"
    elif 112.5 <= deg < 157.5:
        return "西北"
    elif 157.5 <= deg < 202.5:
        return "西"
    elif 202.5 <= deg < 247.5:
        return "西南"
    elif 247.5 <= deg < 292.5:
        return "南"
    else:
        return "东南"


def format_coords(coords: list, time_offset: float = 0.0) -> str:
    """Format coordinate list into readable text."""
    lines = []
    for i, (x, y) in enumerate(coords):
        t = time_offset + i * FRAME_INTERVAL
        lines.append(f"t={t:.1f}s: ({x:.2f}, {y:.2f})")
    return "\n".join(lines)


def sample_to_chat(sample: dict) -> dict:
    """Convert a sample to ms-swift chat format."""
    obs_text = format_coords(sample["obs"])
    pred_text = format_coords(sample["pred"], time_offset=OBS_LENGTH * FRAME_INTERVAL)

    user_msg = (
        f"场景：{sample['scene']}（行人密集区域）\n"
        f"行人历史轨迹（过去{OBS_LENGTH * FRAME_INTERVAL:.1f}秒，每{FRAME_INTERVAL}秒采样）：\n"
        f"{obs_text}\n"
        f"平均速度：{sample['avg_speed']}m/s，主要方向：{sample['direction']}\n\n"
        f"请预测该行人未来{PRED_LENGTH * FRAME_INTERVAL:.1f}秒的轨迹。"
    )

    # Compute analysis based on trajectory
    obs = np.array(sample["obs"])
    pred = np.array(sample["pred"])

    obs_vel = np.diff(obs, axis=0)
    pred_vel = np.diff(pred, axis=0)

    obs_speed = np.mean(np.linalg.norm(obs_vel, axis=1)) / FRAME_INTERVAL
    pred_speed = np.mean(np.linalg.norm(pred_vel, axis=1)) / FRAME_INTERVAL

    speed_change = pred_speed - obs_speed
    if abs(speed_change) < 0.1:
        speed_desc = "速度基本保持稳定"
    elif speed_change > 0:
        speed_desc = f"速度略有增加（约+{speed_change:.2f}m/s）"
    else:
        speed_desc = f"速度略有减缓（约{speed_change:.2f}m/s）"

    obs_dir = np.arctan2(obs[-1, 1] - obs[0, 1], obs[-1, 0] - obs[0, 0])
    pred_dir = np.arctan2(pred[-1, 1] - pred[0, 1], pred[-1, 0] - pred[0, 0])
    dir_change = np.degrees(pred_dir - obs_dir)
    if abs(dir_change) < 10:
        dir_desc = "运动方向基本保持直线"
    elif dir_change > 0:
        dir_desc = f"运动方向向左偏转约{abs(dir_change):.0f}°"
    else:
        dir_desc = f"运动方向向右偏转约{abs(dir_change):.0f}°"

    assistant_msg = (
        f"根据行人的运动趋势分析：\n"
        f"- {speed_desc}\n"
        f"- {dir_desc}\n\n"
        f"预测轨迹：\n"
        f"{pred_text}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_length", type=int, default=OBS_LENGTH)
    parser.add_argument("--pred_length", type=int, default=PRED_LENGTH)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", choices=MODES, default="none",
                        help="agent-centric coordinate normalization mode")
    args = parser.parse_args()
    print(f"Normalization mode: {args.normalize}")

    random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Process train split
    train_dir = RAW_DIR / "train"
    if train_dir.exists():
        for f in sorted(train_dir.glob("*.txt")):
            scene = f.stem
            print(f"Processing train/{scene}...")
            trajectories = parse_eth_ucy(f)
            samples = extract_samples(trajectories, scene, args.obs_length,
                                      args.pred_length, args.normalize)
            print(f"  -> {len(samples)} samples from {len(trajectories)} pedestrians")
            all_samples.extend(samples)

    # Process test split
    test_dir = RAW_DIR / "test"
    test_samples = []
    if test_dir.exists():
        for f in sorted(test_dir.glob("*.txt")):
            scene = f.stem
            print(f"Processing test/{scene}...")
            trajectories = parse_eth_ucy(f)
            samples = extract_samples(trajectories, scene, args.obs_length,
                                      args.pred_length, args.normalize)
            print(f"  -> {len(samples)} samples from {len(trajectories)} pedestrians")
            test_samples.extend(samples)

    if not all_samples and not test_samples:
        print("No samples found! Run download_datasets.py first.")
        return
    if not all_samples:
        print("Note: no training split found; processing test split only.")

    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * args.train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Convert to chat format
    for name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        if not samples:
            continue
        out_path = OUT_DIR / f"trajectory_{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in samples:
                chat = sample_to_chat(sample)
                f.write(json.dumps(chat, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples)} samples to {out_path}")

    # Also save combined for ms-swift
    combined = []
    for sample in train_samples + val_samples:
        combined.append(sample_to_chat(sample))

    combined_path = OUT_DIR / "trajectory_sft.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(combined)} samples to {combined_path}")

    # Print stats
    print(f"\nDataset statistics:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val:   {len(val_samples)}")
    print(f"  Test:  {len(test_samples)}")
    print(f"  Total: {len(train_samples) + len(val_samples) + len(test_samples)}")


if __name__ == "__main__":
    main()

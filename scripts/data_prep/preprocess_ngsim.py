#!/usr/bin/env python3
"""Preprocess NGSIM vehicle trajectories into the ms-swift chat format.

NGSIM (US DOT) is 10 Hz vehicle trajectory data with coordinates in FEET.
This parser:
  - accepts either Socrata lowercase headers (vehicle_id, frame_id, local_x, ...)
    or the classic uppercase headers (Vehicle_ID, Frame_ID, Local_X, ...);
  - converts feet -> meters (x0.3048);
  - downsamples 10 Hz -> 0.4 s (every 4th frame) to match the 8-obs/12-pred window;
  - emits the SAME chat format as the rest of the pipeline (agent=车辆), reusing
    the shared prompt builder, so generate_predictions / evaluate / baselines all
    work unchanged.

Data acquisition: data.transportation.gov blocks datacenter IPs (HTTP 403), so
download the NGSIM CSV from a browser / trusted host and point --csv at it, or
use scripts/data_prep/download_datasets.py --datasets ngsim (which documents the
options). Any CSV with the columns above works.
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from prompt_format import (build_user_prompt, system_prompt, format_coords,  # noqa: E402
                           direction_label, OBS_LENGTH, PRED_LENGTH, FRAME_INTERVAL)
from trajectory_norm import normalize_obs_pred, MODES  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
FEET_TO_M = 0.3048
NGSIM_HZ = 10                       # native sample rate
DOWNSAMPLE = int(round(NGSIM_HZ * FRAME_INTERVAL))   # 10Hz -> 0.4s => every 4th

# column name -> canonical, case-insensitive
COLMAP = {
    "vehicle_id": "vid", "frame_id": "frame",
    "local_x": "x", "local_y": "y",
    "location": "loc",
}


def _resolve_columns(header):
    idx = {}
    lower = {h.lower().strip(): i for i, h in enumerate(header)}
    for name, canon in COLMAP.items():
        if name in lower:
            idx[canon] = lower[name]
    missing = {"vid", "frame", "x", "y"} - set(idx)
    if missing:
        raise ValueError(f"NGSIM CSV missing columns for {missing}; header={header}")
    return idx


def parse_ngsim(csv_path):
    """Return dict[vehicle_id] -> sorted list of (frame, x_m, y_m), plus location."""
    trajs = defaultdict(list)
    location = None
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = _resolve_columns(header)
        has_loc = "loc" in idx
        for row in reader:
            if not row:
                continue
            try:
                vid = int(float(row[idx["vid"]]))
                frame = int(float(row[idx["frame"]]))
                x = float(row[idx["x"]]) * FEET_TO_M
                y = float(row[idx["y"]]) * FEET_TO_M
            except (ValueError, IndexError):
                continue
            trajs[vid].append((frame, x, y))
            if has_loc and location is None and row[idx["loc"]]:
                location = row[idx["loc"]]
    for vid in trajs:
        trajs[vid].sort(key=lambda t: t[0])
    return trajs, (location or "高速公路")


def extract_windows(trajs, obs_len=OBS_LENGTH, pred_len=PRED_LENGTH):
    """Downsample each vehicle to 0.4s and slide obs+pred windows over it."""
    total = obs_len + pred_len
    samples = []
    for vid, tr in trajs.items():
        frames = [t[0] for t in tr]
        coords = {t[0]: (t[1], t[2]) for t in tr}
        # keep frames on the 0.4s grid, requiring the native frames to be contiguous
        base = frames[0]
        grid = [fr for fr in frames if (fr - base) % DOWNSAMPLE == 0]
        for i in range(len(grid) - total + 1):
            window = grid[i:i + total]
            if any(window[k + 1] - window[k] != DOWNSAMPLE for k in range(total - 1)):
                continue
            pts = np.array([coords[fr] for fr in window])
            samples.append(pts)
    return samples


def sample_to_chat(pts, scene, normalize):
    obs, pred = pts[:OBS_LENGTH], pts[OBS_LENGTH:]
    obs, pred, _ = normalize_obs_pred(obs, pred, normalize)
    avg_speed = round(float(np.mean(np.linalg.norm(np.diff(obs, axis=0), axis=1))
                            / FRAME_INTERVAL), 2)
    heading = np.arctan2(obs[-1, 1] - obs[0, 1], obs[-1, 0] - obs[0, 0])
    dir_label = direction_label(heading)
    user_msg = build_user_prompt(obs.tolist(), scene, avg_speed, dir_label,
                                 agent_label="车辆")
    pred_text = format_coords(pred.tolist(), time_offset=OBS_LENGTH * FRAME_INTERVAL)
    assistant = f"根据车辆的运动趋势分析：\n- 保持当前运动趋势\n\n预测轨迹：\n{pred_text}"
    return {"messages": [
        {"role": "system", "content": system_prompt("vehicle")},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant},
    ]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="NGSIM CSV path")
    p.add_argument("--output_file", default=str(OUT_DIR / "ngsim_test.jsonl"))
    p.add_argument("--normalize", choices=MODES, default="none")
    p.add_argument("--scene", default=None, help="override scene label")
    p.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = p.parse_args()

    trajs, location = parse_ngsim(args.csv)
    scene = args.scene or f"高速公路（{location}）"
    samples = extract_windows(trajs)
    if args.max_samples:
        samples = samples[:args.max_samples]

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for pts in samples:
            f.write(json.dumps(sample_to_chat(pts, scene, args.normalize),
                               ensure_ascii=False) + "\n")
    print(f"vehicles={len(trajs)} location={location} "
          f"samples={len(samples)} -> {args.output_file}")


if __name__ == "__main__":
    main()

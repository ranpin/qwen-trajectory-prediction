#!/usr/bin/env python3
"""Generate a synthetic vehicle test set in the eval JSONL schema.

Lets the whole eval chain (evaluate.py / CVM / viz / action_to_traj) run
end-to-end BEFORE PhysicalAI-AV access is granted. Kinematic (unicycle) vehicle
motion at 10 Hz: cruise / lane-change / turn, 5-30 m/s.

Schema per line: {"id","obs":[[x,y]... obs_len],"gt":[[x,y]... 64],"scene"}
"""

import json
import math
import random
import argparse

DT = 0.1  # 10 Hz


def one(obs_len, pred_len, rng):
    scene = rng.choice(["highway", "urban", "intersection", "roundabout"])
    v = rng.uniform(5, 30)
    heading = rng.uniform(0, 2 * math.pi)
    maneuver = rng.choice(["cruise", "lane_change", "turn"])
    total = obs_len + pred_len
    x = y = 0.0
    pts = []
    for k in range(total):
        pts.append([round(x, 3), round(y, 3)])
        if maneuver == "turn":
            yaw = math.radians(rng.uniform(15, 40)) / (total * DT)
        elif maneuver == "lane_change":
            yaw = 0.06 if obs_len <= k < obs_len + 6 else (-0.06 if obs_len + 6 <= k < obs_len + 12 else 0.0)
        else:
            yaw = rng.gauss(0, 0.002)
        heading += yaw * DT * (v if maneuver == "turn" else 1.0)
        v = max(1.0, v + rng.gauss(0, 0.3))
        x += v * math.cos(heading) * DT
        y += v * math.sin(heading) * DT
    return {"id": None, "obs": pts[:obs_len], "gt": pts[obs_len:], "scene": scene}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--obs_len", type=int, default=8)
    ap.add_argument("--pred_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/av_subset/synthetic_test.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for i in range(args.n):
            s = one(args.obs_len, args.pred_len, rng)
            s["id"] = f"syn_{i:04d}"
            f.write(json.dumps(s) + "\n")
    print(f"wrote {args.n} synthetic AV samples -> {args.out}")


if __name__ == "__main__":
    main()

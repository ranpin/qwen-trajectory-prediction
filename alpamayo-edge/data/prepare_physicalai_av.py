#!/usr/bin/env python3
"""M1 — Build an eval subset from NVIDIA PhysicalAI-Autonomous-Vehicles.

The dataset is gated (accept the license on HF) and huge (133 TB) — only pull a
small subset (a few hundred clips) via the official `physical_ai_av` devkit, then
convert each clip's ego trajectory into the eval JSONL schema used by eval/:

  {"id", "obs": [[x,y],...], "gt": [[x,y],... 64], "scene": "..."}

Trajectory spec aligned to Alpamayo: 6.4 s horizon, 64 waypoints @ 10 Hz; the
observation window length is configurable (--obs).

This is a scaffold — the exact egomotion field names come from the devkit
(`pip install physical_ai_av`); wire them at M1 after dataset access is granted.
"""

import argparse
import json
# from physical_ai_av import ...   # TODO(M1): official devkit

PRED_LEN = 64          # 6.4 s @ 10 Hz
PRED_HZ = 10


def clip_to_sample(clip, obs_len):
    """TODO(M1): extract ego (x,y) at 10 Hz from a clip's egomotion labels.

    Return dict(id, obs=obs_len x 2, gt=PRED_LEN x 2, scene) in a local frame
    anchored at the ego position at t=0 (the dataset already uses this frame).
    """
    raise NotImplementedError("wire physical_ai_av egomotion extraction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/av_subset/test.jsonl")
    ap.add_argument("--max_clips", type=int, default=300)
    ap.add_argument("--obs", type=int, default=20, help="observed waypoints")
    ap.parse_args()
    print("TODO(M1): download subset via physical_ai_av, then emit eval JSONL.")
    print("Schema: {'id','obs':[[x,y]...],'gt':[[x,y]...64],'scene'}")


if __name__ == "__main__":
    main()

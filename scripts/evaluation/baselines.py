#!/usr/bin/env python3
"""Physics baselines for trajectory prediction (no model needed).

Produces predictions in the SAME text format the LLM emits, so they can be fed
straight into evaluate.py and compared on identical footing.

- cvm: Constant Velocity Model. Uses the mean per-step velocity over the whole
  observation window and extrapolates linearly. This is the classic strong
  single-mode baseline on ETH/UCY (Schöller et al., 2020) — the key sanity
  check for whether the LLM adds value over trivial extrapolation.
- cv:  Last-step constant velocity. Uses only the final observed step's velocity.

Observation coordinates are parsed from the user prompt (which contains only the
history), reusing evaluate.parse_trajectory (no 预测轨迹 marker → all t= lines).
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Reuse the exact coord formatter and constants used to build training data,
# and the exact parser used at eval time, so formats line up perfectly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_prep"))
from preprocess import format_coords, OBS_LENGTH, PRED_LENGTH, FRAME_INTERVAL  # noqa: E402
from evaluate import parse_trajectory  # noqa: E402

PRED_TIME_OFFSET = OBS_LENGTH * FRAME_INTERVAL  # first predicted timestamp


def _user_prompt(item):
    return next((m["content"] for m in item["messages"] if m["role"] == "user"), "")


def cvm_predict(obs, n_pred, per_step_vel):
    """Extrapolate n_pred steps from the last observed point at a fixed velocity."""
    last = obs[-1]
    return np.array([last + per_step_vel * (k + 1) for k in range(n_pred)])


def predict(obs, model, n_pred):
    if len(obs) < 2:
        return None
    if model == "cvm":
        vel = (obs[-1] - obs[0]) / (len(obs) - 1)      # mean per-step velocity
    elif model == "cv":
        vel = obs[-1] - obs[-2]                          # last-step velocity
    else:
        raise ValueError(f"unknown model: {model}")
    return cvm_predict(obs, n_pred, vel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", choices=["cvm", "cv"], default="cvm")
    parser.add_argument("--pred_length", type=int, default=PRED_LENGTH)
    args = parser.parse_args()

    test_data = [json.loads(l) for l in open(args.test_file)]
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0
    with open(out_path, "w", encoding="utf-8") as out:
        for item in test_data:
            prompt = _user_prompt(item)
            obs = parse_trajectory(prompt)               # no marker → history coords
            if obs is None or len(obs) < 2:
                skipped += 1
                continue
            pred = predict(obs, args.model, args.pred_length)
            pred_text = "预测轨迹：\n" + format_coords(pred.tolist(),
                                                      time_offset=PRED_TIME_OFFSET)
            out.write(json.dumps({"prompt": prompt, "prediction": pred_text},
                                 ensure_ascii=False) + "\n")
            written += 1

    print(f"{args.model}: wrote {written} predictions to {out_path} "
          f"(skipped {skipped})")


if __name__ == "__main__":
    main()

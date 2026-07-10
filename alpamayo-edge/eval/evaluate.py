#!/usr/bin/env python3
"""Evaluate trajectory predictions (Alpamayo / CVM) on an AV test set.

Sample JSONL schema (one object per line):
  {"id": "clip_uuid", "obs": [[x,y],...], "gt": [[x,y],... <=64], "pred": [[x,y],...]}
- "obs":  past waypoints (used for CVM baselines)
- "gt":   ground-truth future waypoints (from PhysicalAI-AV ego trajectory)
- "pred": model prediction (optional; or compute a baseline with --baseline)

Usage:
  # score model predictions already in the file
  python evaluate.py --samples preds.jsonl --label "Alpamayo-R1 INT4" -o out.json
  # score the CVM baseline computed from obs
  python evaluate.py --samples gt.jsonl --baseline cvm --label CVM -o cvm.json
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from metrics import aggregate  # noqa: E402
from baselines import cvm_predict, cv_last_predict  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True, help="JSONL with obs/gt(/pred)")
    p.add_argument("--baseline", choices=["none", "cvm", "cv"], default="none",
                   help="compute prediction from obs instead of using 'pred'")
    p.add_argument("--pred_field", default="pred")
    p.add_argument("--n_pred", type=int, default=64)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    pairs, skipped = [], 0
    for line in open(args.samples):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        gt = s.get("gt")
        if not gt:
            skipped += 1
            continue
        if args.baseline == "cvm":
            pred = cvm_predict(s["obs"], args.n_pred)
        elif args.baseline == "cv":
            pred = cv_last_predict(s["obs"], args.n_pred)
        else:
            pred = s.get(args.pred_field)
            if pred is None:
                skipped += 1
                continue
        pairs.append((np.asarray(pred, float), np.asarray(gt, float)))

    res = aggregate(pairs)
    res["label"] = args.label
    res["skipped"] = skipped

    print(f"\n=== {args.label or args.samples} ===")
    if res.get("num", 0) == 0:
        print(f"No valid samples (skipped {skipped}).")
        return
    print(f"samples={res['num']}  skipped={skipped}")
    print(f"ADE {res['meanADE']:.3f} ± {res['meanADE_ci95']:.3f} m (median {res['medianADE']:.3f})")
    print(f"FDE {res['meanFDE']:.3f} ± {res['meanFDE_ci95']:.3f} m (median {res['medianFDE']:.3f})")
    print(f"Miss Rate (>2m): {res['miss_rate']*100:.1f}%")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump(res, open(args.output, "w"), indent=2, ensure_ascii=False)
        print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()

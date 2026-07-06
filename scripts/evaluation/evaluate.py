#!/usr/bin/env python3
"""Evaluate trajectory prediction on a test set (synthetic or real ETH/UCY).

Metrics are single-prediction ADE/FDE (one trajectory per sample). This is NOT
the best-of-K minADE_K / minFDE_K reported in the multimodal-forecasting
literature — do not compare these numbers against best-of-20 benchmarks.

Predictions are matched to ground truth by prompt text, not by file position,
so a partial predictions file (e.g. only the first N samples) evaluates
correctly instead of silently mis-aligning.
"""

import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Coordinate lines appear after this marker in both GT and model output. Slicing
# on it prevents echoed history coordinates (from the prompt) being parsed as
# predicted points, which would corrupt the GT/pred alignment.
PRED_MARKER = "预测轨迹"


def compute_ade(pred, gt):
    """Average Displacement Error."""
    return np.mean(np.linalg.norm(pred - gt, axis=1))


def compute_fde(pred, gt):
    """Final Displacement Error."""
    return np.linalg.norm(pred[-1] - gt[-1])


def parse_trajectory(text):
    """Extract predicted (x, y) coordinates from model/GT text.

    Only the portion after PRED_MARKER is parsed when the marker is present.
    """
    if PRED_MARKER in text:
        text = text.split(PRED_MARKER, 1)[1]
    coords = []
    for line in text.split('\n'):
        if 't=' in line and '(' in line and ')' in line:
            try:
                start = line.index('(')
                end = line.index(')')
                x, y = map(float, line[start + 1:end].split(','))
                coords.append([x, y])
            except (ValueError, IndexError):
                continue
    return np.array(coords) if coords else None


def _user_prompt(test_item):
    """The user message that was sent to the model for this test item."""
    return next((m['content'] for m in test_item['messages']
                 if m['role'] == 'user'), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test data JSONL file")
    parser.add_argument("--predictions_file", type=str, required=True,
                        help="Path to model predictions JSONL file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--label", type=str, default=None,
                        help="Optional label recorded in the results JSON")
    args = parser.parse_args()

    test_data = [json.loads(l) for l in open(args.test_file)]
    predictions = [json.loads(l) for l in open(args.predictions_file)]

    # Index ground truth by the exact prompt so a partial predictions file still
    # aligns correctly (predictions may cover only a subset of the test set).
    gt_by_prompt = {}
    for item in test_data:
        prompt = _user_prompt(item)
        if prompt is not None:
            gt_by_prompt[prompt] = item['messages'][-1]['content']

    ades, fdes = [], []
    unmatched = 0          # prediction whose prompt isn't in the test set
    parse_fail = 0         # GT or prediction produced no parseable coords
    len_mismatch = 0       # GT and prediction had different point counts

    for pred_item in tqdm(predictions, desc="Evaluating"):
        prompt = pred_item.get('prompt')
        gt_text = gt_by_prompt.get(prompt)
        if gt_text is None:
            unmatched += 1
            continue

        gt_coords = parse_trajectory(gt_text)
        pred_coords = parse_trajectory(pred_item.get('prediction', ''))
        if gt_coords is None or pred_coords is None:
            parse_fail += 1
            continue

        if len(gt_coords) != len(pred_coords):
            len_mismatch += 1
        min_len = min(len(gt_coords), len(pred_coords))
        if min_len < 2:
            parse_fail += 1
            continue
        gt_coords, pred_coords = gt_coords[:min_len], pred_coords[:min_len]

        ades.append(compute_ade(pred_coords, gt_coords))
        fdes.append(compute_fde(pred_coords, gt_coords))

    if not ades:
        print("No valid predictions found! "
              f"(unmatched={unmatched}, parse_fail={parse_fail})")
        return

    ades, fdes = np.array(ades), np.array(fdes)
    # 95% CI of the mean via normal approximation (n is large enough here).
    ade_ci = 1.96 * ades.std(ddof=1) / np.sqrt(len(ades))
    fde_ci = 1.96 * fdes.std(ddof=1) / np.sqrt(len(fdes))

    results = {
        "label": args.label,
        "num_predictions": len(predictions),
        "num_evaluated": int(len(ades)),
        "unmatched": unmatched,
        "parse_failures": parse_fail,
        "length_mismatches": len_mismatch,
        "metric_note": "single-prediction ADE/FDE, not best-of-K minADE",
        "meanADE": float(ades.mean()),
        "meanADE_ci95": float(ade_ci),
        "medianADE": float(np.median(ades)),
        "meanFDE": float(fdes.mean()),
        "meanFDE_ci95": float(fde_ci),
        "medianFDE": float(np.median(fdes)),
        "minADE": float(ades.min()),
        "maxADE": float(ades.max()),
        "stdADE": float(ades.std(ddof=1)),
        "minFDE": float(fdes.min()),
        "maxFDE": float(fdes.max()),
        "stdFDE": float(fdes.std(ddof=1)),
        "miss_rate_2m": float(np.mean(fdes > 2.0)),
    }

    print("\n=== Evaluation Results ===")
    if args.label:
        print(f"Label: {args.label}")
    print(f"Predictions: {results['num_predictions']}  "
          f"Evaluated: {results['num_evaluated']}  "
          f"Unmatched: {unmatched}  Parse-fail: {parse_fail}  "
          f"Len-mismatch: {len_mismatch}")
    print(f"\nADE mean: {results['meanADE']:.4f} ± {ade_ci:.4f} m (95% CI), "
          f"median {results['medianADE']:.4f} m")
    print(f"FDE mean: {results['meanFDE']:.4f} ± {fde_ci:.4f} m (95% CI), "
          f"median {results['medianFDE']:.4f} m")
    print(f"Miss Rate (>2m): {results['miss_rate_2m'] * 100:.2f}%")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

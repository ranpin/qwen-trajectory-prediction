#!/usr/bin/env python3
"""Trajectory metrics (ADE / FDE / Miss Rate) for the Alpamayo-edge project.

Migrated from the parent repo's scripts/evaluation/evaluate.py, but operating on
numeric trajectory arrays (N, 2) instead of parsing text prompts — Alpamayo
outputs waypoints directly (6.4s horizon, 64 waypoints @ 10 Hz), so there is no
LLM text to parse here.
"""

import numpy as np

MISS_THRESHOLD_M = 2.0  # FDE > this counts as a miss


def ade(pred, gt):
    """Average Displacement Error over the overlapping horizon."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    n = min(len(pred), len(gt))
    return float(np.mean(np.linalg.norm(pred[:n] - gt[:n], axis=1)))


def fde(pred, gt):
    """Final Displacement Error at the last common waypoint."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    n = min(len(pred), len(gt))
    return float(np.linalg.norm(pred[n - 1] - gt[n - 1]))


def aggregate(pairs):
    """Aggregate metrics over an iterable of (pred, gt) arrays.

    Returns a dict with mean ADE/FDE + 95% CI, medians, and miss rate.
    """
    ades = np.array([ade(p, g) for p, g in pairs])
    fdes = np.array([fde(p, g) for p, g in pairs])
    if len(ades) == 0:
        return {"num": 0}

    def ci95(x):
        return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

    return {
        "num": int(len(ades)),
        "meanADE": float(ades.mean()), "meanADE_ci95": ci95(ades),
        "medianADE": float(np.median(ades)),
        "meanFDE": float(fdes.mean()), "meanFDE_ci95": ci95(fdes),
        "medianFDE": float(np.median(fdes)),
        "miss_rate": float(np.mean(fdes > MISS_THRESHOLD_M)),
        "minADE": float(ades.min()), "maxADE": float(ades.max()),
    }

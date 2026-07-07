"""Coordinate normalization shared by data prep and the demo.

The model predicts absolute coordinates. On real ETH/UCY those are large world
positions (e.g. 13.6, 5.8) far from the synthetic training distribution, which
hurts accuracy. Normalizing each trajectory into an agent-centric frame removes
that translation/rotation nuisance so the model only has to learn *relative*
motion.

The transform is computed from the OBSERVATION only (causal) and applied to both
observation and future, so it is invertible and leaks no future information. It
is fully determined by the observation, so the demo recomputes it at inference
time rather than persisting it.

Modes:
- none:             identity.
- translate:        last observed point -> origin.
- translate_rotate: last observed point -> origin, and the overall observed
                    heading (obs[-1]-obs[0]) rotated onto +x (agent-centric).
"""

import numpy as np

MODES = ("none", "translate", "translate_rotate")
_EPS = 1e-6


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def compute_transform(obs, mode):
    """Return (origin, angle) describing the normalization for this observation.

    angle is the heading (radians) that gets rotated onto +x; 0 for non-rotating
    modes or when the observation has no net displacement.
    """
    obs = np.asarray(obs, dtype=float)
    if mode == "none":
        return np.zeros(2), 0.0
    origin = obs[-1].copy()
    if mode == "translate":
        return origin, 0.0
    if mode == "translate_rotate":
        disp = obs[-1] - obs[0]
        angle = float(np.arctan2(disp[1], disp[0])) if np.linalg.norm(disp) > _EPS else 0.0
        return origin, angle
    raise ValueError(f"unknown normalize mode: {mode!r} (expected one of {MODES})")


def apply_transform(coords, origin, angle):
    """World -> normalized frame."""
    coords = np.asarray(coords, dtype=float)
    shifted = coords - origin
    if abs(angle) < _EPS:
        return shifted
    return shifted @ _rot(-angle).T


def invert_transform(coords, origin, angle):
    """Normalized frame -> world (inverse of apply_transform)."""
    coords = np.asarray(coords, dtype=float)
    if abs(angle) < _EPS:
        return coords + origin
    return coords @ _rot(angle).T + origin


def normalize_obs_pred(obs, pred, mode):
    """Normalize an (obs, pred) pair with one shared, observation-derived transform.

    Returns (obs_n, pred_n, (origin, angle)).
    """
    origin, angle = compute_transform(obs, mode)
    return (apply_transform(obs, origin, angle),
            apply_transform(pred, origin, angle),
            (origin, angle))

#!/usr/bin/env python3
"""Constant-Velocity (CVM) baseline for vehicle trajectory prediction.

Migrated from the parent repo's scripts/evaluation/baselines.py. CVM is the
classic strong single-mode baseline — the model must beat it to justify its cost.
Operates on numeric arrays (agnostic to how predictions are produced).
"""

import numpy as np


def cvm_predict(obs, n_pred=64):
    """Extrapolate n_pred waypoints at the mean per-step velocity of the obs.

    obs: (T, 2) past waypoints (uniform time step). Returns (n_pred, 2).
    """
    obs = np.asarray(obs, float)
    if len(obs) < 2:
        return np.repeat(obs[-1:], n_pred, axis=0)
    vel = (obs[-1] - obs[0]) / (len(obs) - 1)          # mean per-step velocity
    return np.array([obs[-1] + vel * (k + 1) for k in range(n_pred)])


def cv_last_predict(obs, n_pred=64):
    """Extrapolate using only the last observed step's velocity."""
    obs = np.asarray(obs, float)
    if len(obs) < 2:
        return np.repeat(obs[-1:], n_pred, axis=0)
    vel = obs[-1] - obs[-2]
    return np.array([obs[-1] + vel * (k + 1) for k in range(n_pred)])

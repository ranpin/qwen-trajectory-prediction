#!/usr/bin/env python3
"""Convert Alpamayo action output → (x, y) waypoints for evaluation.

Alpamayo's `output_trajectory` is a list of (accel, kappa) = (acceleration,
curvature) pairs, NOT (x, y). To score ADE/FDE against ground-truth positions we
integrate a kinematic (unicycle) model from the ego's current speed/heading:

  v_{t+1}  = max(0, v_t + a_t * dt)
  heading += v_t * kappa_t * dt          # curvature kappa = yaw_rate / v
  x += v*cos(heading)*dt ; y += v*sin(heading)*dt

v0/heading0 come from the last observed step. dt = 0.1s (10 Hz), 64 steps = 6.4s.
"""

import math
import numpy as np


def estimate_v0_heading(obs, dt=0.1):
    """Initial speed (m/s) and heading (rad) from the last observed segment."""
    obs = np.asarray(obs, float)
    d = obs[-1] - obs[-2]
    return float(np.linalg.norm(d) / dt), float(math.atan2(d[1], d[0]))


def action_to_xy(accel, kappa, v0, heading0=0.0, dt=0.1, origin=(0.0, 0.0)):
    x, y = origin
    v, h = v0, heading0
    pts = []
    for a, k in zip(accel, kappa):
        v = max(0.0, v + a * dt)
        h = h + v * k * dt
        x += v * math.cos(h) * dt
        y += v * math.sin(h) * dt
        pts.append([x, y])
    return np.array(pts)


def alpamayo_output_to_xy(output_trajectory, obs, dt=0.1):
    """output_trajectory: [[accel, kappa], ...]; anchor at last observed point."""
    ot = np.asarray(output_trajectory, float)
    v0, h0 = estimate_v0_heading(obs, dt)
    return action_to_xy(ot[:, 0], ot[:, 1], v0, h0, dt, origin=tuple(np.asarray(obs, float)[-1]))

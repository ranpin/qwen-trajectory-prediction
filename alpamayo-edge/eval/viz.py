#!/usr/bin/env python3
"""BEV trajectory visualization (observed / predicted / CVM / ground truth).

Migrated from the parent repo's demo/app.py plotting. Renders a bird's-eye-view
PNG comparing model prediction against the CVM baseline and ground truth.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# CJK-capable font when available (avoids tofu in titles).
_avail = {f.name for f in fm.fontManager.ttflist}
for _n in ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "SimHei"]:
    if _n in _avail:
        plt.rcParams["font.sans-serif"] = [_n]
        break
plt.rcParams["axes.unicode_minus"] = False


def plot_bev(obs, series, out_png, title="Trajectory (BEV)"):
    """obs: (T,2); series: dict label->(N,2) e.g. {'LLM':..,'CVM':..,'GT':..}."""
    styles = {"LLM": "r--^", "CVM": "g:s", "GT": "k-.", "Observed": "b-o"}
    fig, ax = plt.subplots(figsize=(9, 8))
    if obs is not None and len(obs):
        obs = np.asarray(obs, float)
        ax.plot(obs[:, 0], obs[:, 1], styles["Observed"], lw=2, ms=5,
                label="Observed", alpha=0.9)
    for label, arr in (series or {}).items():
        if arr is None or not len(arr):
            continue
        arr = np.asarray(arr, float)
        st = styles.get(label, "-")
        if obs is not None and len(obs):
            ax.plot([obs[-1, 0], arr[0, 0]], [obs[-1, 1], arr[0, 1]],
                    st[0] + ":", lw=1, alpha=0.4)
        ax.plot(arr[:, 0], arr[:, 1], st, lw=2, ms=4, label=label, alpha=0.9)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(title); ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png

#!/usr/bin/env python3
"""Model-size vs precision Pareto: accuracy against speed and against footprint.

All four points are measured on real hardware, all accuracies on the same 210-item
Cosmos-Reason1-Benchmark subset with the committed scorer
(eval/accuracy/benchmark/score_mc.py, which reproduces the historical numbers exactly).

The headline this figure exists to make: quantization costs nothing measurable
(McNemar p > 0.4) while shrinking the model 8B -> 2B costs 11 points and IS significant
(p = 0.005). So on this hardware you quantize aggressively and keep the big model.

Run:  .venv/bin/python alpamayo-edge/eval/pareto_viz.py
Out:  alpamayo-edge/docs/figures/pareto.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_style import apply, bar_kw, ANNOT, C4, C8, CF, RULE, WIDTH_FULL

apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "..", "docs", "figures")

# name, engine GB, decode tok/s, accuracy %, Wilson CI, McNemar p vs FP16, colour, device
POINTS = [
    ("FP16 8B",  15.15, 11.2, 76.2, (70.0, 81.4), None,  CF,   "goat"),
    ("INT8 8B",   8.30, 19.7, 75.2, (69.0, 80.6), 0.845, C8,   "dog"),
    ("INT4 8B",   4.85, 30.9, 73.8, (67.5, 79.3), 0.424, C4,   "dog"),
    ("INT4 2B",   1.36, 74.6, 65.2, (58.6, 71.4), 0.005, RULE, "dog"),
]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 4.4))

for ax, xs, xlabel, title in (
        (a1, [p[2] for p in POINTS], "decode throughput (tokens / s)",
         "Accuracy vs speed"),
        (a2, [p[1] for p in POINTS], "LLM engine size (GB)",
         "Accuracy vs footprint")):
    ys = [p[3] for p in POINTS]
    for (name, gb, tps, acc, ci, p, col, dev), x in zip(POINTS, xs):
        yerr = [[acc - ci[0]], [ci[1] - acc]]
        ax.errorbar([x], [acc], yerr=yerr, fmt="o", color=col, ms=7, capsize=3,
                    elinewidth=0.9, capthick=0.9, zorder=5)
        lab = name if p is None else f"{name}\np={p:.3f}"
        ax.annotate(lab, (x, acc), xytext=(8, -4), textcoords="offset points",
                    fontsize=ANNOT, color=col, va="top")
    # connect the 8B precision sweep to show it is flat
    e8 = [(x, y) for (x, y), p in zip(zip(xs, ys), POINTS) if p[0].endswith("8B")]
    ax.plot([x for x, _ in e8], [y for _, y in e8], "-", color="#999999", lw=0.9,
            zorder=1, label="8B precision sweep (flat)")
    ax.axhline(76.2, color=CF, ls="--", lw=0.9, zorder=1)
    ax.set_xlabel(xlabel); ax.set_title(title)
    ax.set_ylim(52, 90)
    ax.grid(True, axis="both")

a1.set_ylabel("multiple-choice accuracy (%)  [n = 210]")
a1.annotate("FP16 reference 76.2%", (0.99, 76.2), xycoords=("axes fraction", "data"),
            ha="right", va="bottom", fontsize=ANNOT, color=CF)
a2.set_xscale("log")
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Precision vs model size (Jetson Orin, MAXN, batch = 1; error bars = Wilson 95% CI)")
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(os.path.join(OUT, "pareto.png")); plt.close(fig)

print(f"{'config':10} {'GB':>6} {'tok/s':>7} {'acc%':>6} {'vs FP16':>8} {'McNemar':>8}")
for name, gb, tps, acc, ci, p, col, dev in POINTS:
    print(f"{name:10} {gb:6.2f} {tps:7.1f} {acc:6.1f} {acc-76.2:+8.1f} "
          f"{'ref' if p is None else f'{p:.3f}':>8}  ({dev})")
print("\nquantization 8B: 76.2 -> 73.8 over a 3.1x engine shrink and 2.8x decode speedup, "
      "all p > 0.4 (not significant)")
print("model shrink 8B->2B at INT4: 73.8 -> 65.2 (-8.6 pts vs INT4 8B, -11.0 vs FP16), "
      "p = 0.005 (significant)")
print("wrote", os.path.join(OUT, "pareto.png"))

#!/usr/bin/env python3
"""W4A16 GEMV kernel study: hand-written variants vs TRT vs the real DRAM ceiling.

Data: eval/kernels/results.csv, produced by w4a16_gemv.cu on orin-dog
(nvcc -O3 -arch=sm_87). The point of the benchmark is the CEILING measurement:
"TRT reaches 72% of the 204.8 GB/s theoretical peak" sounds like there is 28%
to win, but this Orin only delivers ~152 GB/s of streaming read bandwidth, so
TRT is actually at ~97% of what the hardware can do.

Run:  .venv/bin/python alpamayo-edge/eval/kernels/gemv_bench_viz.py
Out:  alpamayo-edge/docs/figures/gemv_kernel_bench.png
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from paper_style import apply, bar_kw, ANNOT, C4, C8, CF, RULE, WIDTH_FULL

apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "docs", "figures")
PEAK = 204.8          # theoretical
TRT = 147.9           # TRT-Edge-LLM gemv_kernel, from Nsight (see eval/profile/)

rows = list(csv.DictReader(open(os.path.join(HERE, "results.csv"))))
ceiling = float([r for r in rows if r["kind"] == "ceiling"][0]["GB_per_s"])

shapes, variants = [], []
for r in rows:
    if r["kind"] != "gemv":
        continue
    sh = r["shape"].replace("  ", " ").strip()
    if sh not in shapes:
        shapes.append(sh)
    if r["variant"] not in variants:
        variants.append(r["variant"])
val = {(r["shape"].replace("  ", " ").strip(), r["variant"]): float(r["GB_per_s"])
       for r in rows if r["kind"] == "gemv"}

LABEL = {"v1_scalar": "v1 scalar 32-bit loads",
         "v2_vec128": "v2 128-bit loads",
         "v3_vec128_mlp2": "v3 128-bit + 2 in flight",
         "v4_shmem_half2": "v4 + x in shared mem + half2"}
COLS = ["#adb5bd", "#868e96", "#6c757d", C4]

fig, ax = plt.subplots(figsize=(WIDTH_FULL, 4.4))
x = np.arange(len(shapes))
w = 0.2
for i, v in enumerate(variants):
    vals = [val[(s, v)] for s in shapes]
    bars = ax.bar(x + (i - 1.5) * w, vals, w, color=COLS[i],
                  label=LABEL.get(v, v), **bar_kw())
    for r, vv in zip(bars, vals):
        ax.annotate(f"{vv:.0f}", (r.get_x() + r.get_width() / 2, vv), ha="center",
                    va="bottom", fontsize=ANNOT, xytext=(0, 1.5),
                    textcoords="offset points")

ax.axhline(PEAK, color="#555555", ls=":", lw=1.0)
ax.annotate(f"theoretical peak {PEAK:.1f} GB/s", (0.99, PEAK), xycoords=("axes fraction", "data"),
            ha="right", va="bottom", fontsize=ANNOT, color="#555555")
ax.axhline(ceiling, color=CF, ls="-", lw=1.2)
ax.annotate(f"measured achievable read ceiling {ceiling:.1f} GB/s ({100*ceiling/PEAK:.0f}% of peak)",
            (0.99, ceiling), xycoords=("axes fraction", "data"), ha="right", va="bottom",
            fontsize=ANNOT, color=CF)
ax.axhline(TRT, color=RULE, ls="--", lw=1.2)
ax.annotate(f"TRT-Edge-LLM gemv {TRT:.1f} GB/s = {100*TRT/ceiling:.0f}% of achievable",
            (0.99, TRT), xycoords=("axes fraction", "data"), ha="right", va="top",
            fontsize=ANNOT, color=RULE)

ax.set_xticks(x)
ax.set_xticklabels([s.replace(" (", "\n(") for s in shapes])
ax.set_ylabel("achieved bandwidth (GB/s)")
ax.set_ylim(0, PEAK * 1.12)
ax.set_title("Hand-written W4A16 GEMV vs TensorRT-Edge-LLM (orin-dog, sm_87, MAXN)")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(OUT, "gemv_kernel_bench.png")); plt.close(fig)

best = max(val[(s, v)] for s in shapes for v in variants)
print(f"achievable ceiling   {ceiling:6.1f} GB/s ({100*ceiling/PEAK:.1f}% of theoretical)")
print(f"TRT gemv             {TRT:6.1f} GB/s ({100*TRT/ceiling:.1f}% of achievable)")
print(f"my best variant      {best:6.1f} GB/s ({100*best/ceiling:.1f}% of achievable, "
      f"{TRT/best:.2f}x slower than TRT)")
print(f"max possible gain from a perfect GEMV: {100*(ceiling-TRT)/TRT:.1f}% on this kernel, "
      f"~{100*(ceiling-TRT)/TRT*0.721:.1f}% end-to-end (GEMV is 72.1% of decode)")
print("wrote", os.path.join(OUT, "gemv_kernel_bench.png"))

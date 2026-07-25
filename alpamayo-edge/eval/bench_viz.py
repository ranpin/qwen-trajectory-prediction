#!/usr/bin/env python3
"""Benchmark visualization for Cosmos-Reason2-8B INT4/INT8 on Jetson Orin.

Renders the measured Orin metrics (see docs/edge_deploy_status.md) with the
shared conference-paper style in paper_style.py. Numbers are the actual
measurements, hardcoded here so the figures regenerate deterministically
without a live Orin.

Titles are short and neutral (quantity + measurement conditions); results and
interpretation live in the figure captions on the presentation page, not inside
the images.

Run:  .venv/bin/python alpamayo-edge/eval/bench_viz.py
Out:  alpamayo-edge/docs/figures/*.png
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply, bar_kw, fig_legend, label_bars, ANNOT, C4, C8,
                         CF, RULE, WIDTH_FULL, WIDTH_HALF)

apply()
import matplotlib.pyplot as plt  # noqa: E402  (after apply(), style is set)

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

COND_GOAT = "orin-goat, MAXN, batch = 1"
COND_DOG = "orin-dog, MAXN, batch = 1"


def _bars(ax, labels, v4, v8, ylabel, title, fmt="{:.1f}"):
    """Two-series (INT4 / INT8) grouped bars."""
    x = np.arange(len(labels)); w = 0.38
    b4 = ax.bar(x - w / 2, v4, w, label="INT4 (AWQ)", color=C4, **bar_kw())
    b8 = ax.bar(x + w / 2, v8, w, label="INT8 (SmoothQuant)", color=C8, **bar_kw())
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title)
    for bars in (b4, b8):
        label_bars(ax, bars, fmt)
    if len(labels) == 1:
        ax.set_xlim(-0.75, 0.75)
    ax.margins(y=0.18)


def _bars3(ax, labels, vF, v8, v4, ylabel, title, fmt="{:.1f}", logy=False):
    """Three-series (FP16 / INT8 / INT4) grouped bars."""
    x = np.arange(len(labels)); w = 0.26
    for off, vals, c, lab in [(-w, vF, CF, "FP16"), (0, v8, C8, "INT8 (SQ)"),
                              (w, v4, C4, "INT4 (AWQ)")]:
        bars = ax.bar(x + off, vals, w, label=lab, color=c, **bar_kw())
        label_bars(ax, bars, fmt)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if len(labels) == 1:
        ax.set_xlim(-0.75, 0.75)
    ax.margins(y=0.20)


# ---- Fig 1: throughput, 3-way incl FP16 (goat, same device) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_HALF, 3.4))
_bars3(a1, ["Prefill (512 tok)"], [1718], [3214], [1681], "tokens / s",
       "Prefill", fmt="{:.0f}")
_bars3(a2, ["Decode (per token)"], [11.2], [19.1], [29.7], "tokens / s",
       "Decode", fmt="{:.1f}")
fig.suptitle(f"Throughput ({COND_GOAT})")
fig_legend(fig, a1)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(os.path.join(OUT, "bench_throughput.png")); plt.close(fig)

# ---- Fig 2: power & energy, 3-way incl FP16 (goat) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_HALF, 3.4))
_bars3(a1, ["Prefill", "Decode"], [57.1, 26.7], [51.8, 21.2], [62.6, 22.7],
       "watts (module, 3 rails)", "Module power", fmt="{:.1f}")
_bars3(a2, ["Prefill", "Decode"], [30.1, 0.42], [62.0, 0.90], [26.9, 1.31],
       "tokens / joule", "Energy efficiency (log scale)", fmt="{:.2f}", logy=True)
fig.suptitle(f"Power and energy efficiency ({COND_GOAT})")
fig_legend(fig, a1)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(os.path.join(OUT, "bench_power_energy.png")); plt.close(fig)

# ---- Fig 3: decode throughput vs context length ----
fig, ax = plt.subplots(figsize=(WIDTH_HALF, 3.4))
kv = [128, 512, 1024, 2048, 4000]; tps = [32.1, 31.8, 31.3, 30.5, 29.2]
ax.plot(kv, tps, "-o", color=C4, lw=1.4, ms=4)
for xk, yk in zip(kv, tps):
    ax.annotate(f"{yk}", (xk, yk), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=ANNOT)
ax.set_xlabel("past-KV length (context tokens)")
ax.set_ylabel("decode tokens / s")
ax.set_title(f"INT4 decode throughput vs. context length\n({COND_DOG})")
ax.set_ylim(0, 36)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "decode_scaling.png")); plt.close(fig)

# ---- Fig 4: LLM engine footprint incl FP16 ----
fig, ax = plt.subplots(figsize=(WIDTH_HALF, 3.4))
labs = ["FP16", "INT8 (SQ)", "INT4 (AWQ)"]; sz = [15.15, 8.3, 4.8]
cols = [CF, C8, C4]
b = ax.bar(labs, sz, color=cols, width=0.58, **bar_kw())
for r, v, pc in zip(b, sz, ["100%", "55%", "32%"]):
    ax.annotate(f"{v:.1f} GB\n({pc})", (r.get_x() + r.get_width() / 2, v),
                ha="center", va="bottom", fontsize=ANNOT,
                xytext=(0, 2), textcoords="offset points")
ax.axhline(30, color=RULE, lw=1.0, ls="--")
ax.annotate("orin-dog unified memory ≈ 30 GB", (2.42, 30), color=RULE,
            fontsize=ANNOT, va="bottom", ha="right")
ax.set_ylabel("LLM engine size (GB)"); ax.set_ylim(0, 34)
ax.set_title("LLM engine size by precision")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "footprint.png")); plt.close(fig)

# ---- Fig 5: quantization accuracy drop-off vs FP16 (eval/accuracy/) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 4.25))
# perplexity under the FP16 model (lower = closer to FP16 distribution)
_bars(a1, ["mean PPL"], [1.359], [1.460], "perplexity (FP16 = 1.237)",
      "Perplexity under the FP16 model", fmt="{:.3f}")
a1.axhline(1.237, color=CF, lw=1.0, ls="--")
a1.text(0.015, 1.237, "FP16 reference 1.237", color=CF, fontsize=ANNOT,
        ha="left", va="bottom",
        transform=a1.get_yaxis_transform())
# token agreement with FP16 greedy path
_bars(a2, ["prefix\nagreement %", "prefix len\n(tok)"], [16.4, 20.1], [5.7, 7.2],
      "% / tokens", "Greedy-prefix agreement with FP16", fmt="{:.1f}")
fig.suptitle("Fidelity to FP16 (12 driving / reasoning prompts, greedy decoding)")
fig_legend(fig, a1, ncol=2)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(os.path.join(OUT, "accuracy_dropoff.png")); plt.close(fig)

# ---- Fig 6: task accuracy per subset + overall (Cosmos-Reason1 MC, ground-truth) ----
# Wilson 95% CI error bars. McNemar vs FP16: INT8 p=0.85, INT4 p=0.42 (not significant).
# wilson_ci() reproduces the overall intervals stored in
# eval/accuracy/benchmark/RESULTS.json exactly (FP16 [70.0,81.4], INT8 [69.0,80.6],
# INT4 [67.5,79.3]), so the per-subset intervals it derives are consistent with them.
fig, ax = plt.subplots(figsize=(WIDTH_FULL, 4.25))
groups = ["robovqa\n(n=110, easier)", "robofail\n(n=100, harder)", "ALL\n(n=210)"]
fp16 = [88.2, 63.0, 76.2]; int8 = [87.3, 62.0, 75.2]; int4 = [87.3, 59.0, 73.8]
NS = [110, 100, 210]


def wilson_ci(pct, n, z=1.959963985):
    """Wilson score 95% interval as (lower_err, upper_err) in accuracy points."""
    p = round(pct / 100 * n) / n           # recover the integer correct count
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return pct - 100 * (centre - half), 100 * (centre + half) - pct


x = np.arange(len(groups)); w = 0.26
for off, vals, c, lab in [(-w, fp16, CF, "FP16 (ref)"),
                          (0, int8, C8, "INT8 (SmoothQuant)"),
                          (w, int4, C4, "INT4 (AWQ)")]:
    err = np.array([wilson_ci(v, n) for v, n in zip(vals, NS)]).T
    bars = ax.bar(x + off, vals, w, color=c, label=lab, yerr=err, capsize=2.5,
                  error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor="#333333"),
                  **bar_kw())
    # value labels sit above the upper CI cap, not on top of it
    for r, v, up in zip(bars, vals, err[1]):
        ax.annotate(f"{v:.1f}", (r.get_x() + r.get_width() / 2, v + up),
                    ha="center", va="bottom", fontsize=ANNOT,
                    xytext=(0, 2), textcoords="offset points")
# FP16 overall reference line (the caption points at this); label goes in the
# empty gap between the robofail and ALL groups so it clears every bar label.
ax.axhline(76.2, color=CF, lw=1.0, ls="--", zorder=1)
ax.text(1.5, 76.2, "FP16 overall 76.2%", color=CF, fontsize=ANNOT,
        ha="center", va="bottom")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("multiple-choice accuracy (%)"); ax.set_ylim(40, 100)
ax.set_title("Multiple-choice accuracy on Cosmos-Reason1-Benchmark (n = 210)")
ax.margins(y=0.12)
fig_legend(fig, ax)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(OUT, "benchmark_accuracy.png")); plt.close(fig)

# ---- Fig 7: serving metrics + quantization speedup vs FP16 (all on goat, same device) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 4.25))
# left: TTFT (prefill) & TPOT (decode) grouped bars for FP16/INT8/INT4
metrics = ["TTFT\n(prefill 512)", "TPOT\n(decode /token)"]
fp16 = [298.1, 89.27]; int8 = [159.3, 52.43]; int4 = [304.6, 33.62]
x = np.arange(2); w = 0.26
for off, vals, c, lab in [(-w, fp16, CF, "FP16"), (0, int8, C8, "INT8 (SQ)"),
                          (w, int4, C4, "INT4 (AWQ)")]:
    bars = a1.bar(x + off, vals, w, color=c, label=lab, **bar_kw())
    label_bars(a1, bars, "{:.0f}")
a1.set_xticks(x); a1.set_xticklabels(metrics); a1.set_ylabel("latency (ms)")
a1.set_title("TTFT and TPOT")
a1.margins(y=0.18)
# right: E2E (512-in + 128-out) for the three
labels = ["FP16", "INT8 (SQ)", "INT4 (AWQ)"]; e2e = [11.72, 6.87, 4.61]
cols = [CF, C8, C4]
b = a2.bar(labels, e2e, color=cols, width=0.6, **bar_kw())
for r, v, sp in zip(b, e2e, ["1.0× (ref)", "1.7×", "2.5×"]):
    a2.annotate(f"{v:.1f} s\n{sp}", (r.get_x() + r.get_width() / 2, v),
                ha="center", va="bottom", fontsize=ANNOT,
                xytext=(0, 2), textcoords="offset points")
a2.set_ylabel("E2E latency (s)"); a2.set_ylim(0, 14)
a2.set_title("E2E latency (512-in + 128-out)")
a2.margins(y=0.15)
fig.suptitle(f"Serving metrics ({COND_GOAT}; INT4/INT8 within 3% of orin-dog)")
fig_legend(fig, a1)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(os.path.join(OUT, "serving_metrics.png")); plt.close(fig)

print("wrote:", ", ".join(sorted(os.listdir(OUT))))

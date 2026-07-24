#!/usr/bin/env python3
"""Benchmark visualization for Cosmos-Reason2-8B INT4/INT8 on Jetson Orin.

Renders portfolio-grade charts from the measured Orin metrics (see
docs/edge_deploy_status.md). Numbers are the actual measurements, hardcoded here
so the figures regenerate deterministically without a live Orin.

Run:  .venv/bin/python alpamayo-edge/eval/bench_viz.py
Out:  alpamayo-edge/docs/figures/*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

C4, C8 = "#2563eb", "#f59e0b"  # INT4 blue, INT8 amber
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 130})


def _bars(ax, labels, v4, v8, ylabel, title, fmt="{:.1f}"):
    x = np.arange(len(labels)); w = 0.38
    b4 = ax.bar(x - w/2, v4, w, label="INT4 (AWQ)", color=C4)
    b8 = ax.bar(x + w/2, v8, w, label="INT8 (SmoothQuant)", color=C8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    for bars in (b4, b8):
        for r in bars:
            h = r.get_height()
            ax.annotate(fmt.format(h), (r.get_x()+r.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9,
                        xytext=(0, 2), textcoords="offset points")
    ax.legend(fontsize=9); ax.margins(y=0.18)


# ---- Fig 1: throughput (the core tradeoff) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
_bars(a1, ["Prefill\n(512 tok)"], [1701], [3252], "tokens / sec",
      "Prefill throughput  ->  INT8 wins (1.9x)", fmt="{:.0f}")
_bars(a2, ["Decode\n(per token)"], [30.9], [19.7], "tokens / sec",
      "Decode throughput  ->  INT4 wins (1.57x)", fmt="{:.1f}")
fig.suptitle("Cosmos-Reason2-8B on Jetson Orin (sm_87, MAXN): prefill favors INT8, decode favors INT4",
             fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(OUT, "bench_throughput.png")); plt.close(fig)

# ---- Fig 2: power & energy efficiency ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
_bars(a1, ["Prefill", "Decode"], [61.8, 39.6], [52.1, 40.6], "Watts (module, 3 rails)",
      "Total power draw", fmt="{:.1f}")
_bars(a2, ["Prefill", "Decode"], [27.5, 0.82], [62.8, 0.50], "tokens / Joule",
      "Energy efficiency (log)", fmt="{:.2f}")
a2.set_yscale("log")
fig.suptitle("Power & energy: INT8 2.3x more efficient at prefill, INT4 1.64x at decode",
             fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(OUT, "bench_power_energy.png")); plt.close(fig)

# ---- Fig 3: decode throughput vs context length ----
fig, ax = plt.subplots(figsize=(7, 4.2))
kv = [128, 512, 1024, 2048, 4000]; tps = [32.1, 31.8, 31.3, 30.5, 29.2]
ax.plot(kv, tps, "-o", color=C4, lw=2, ms=6)
for xk, yk in zip(kv, tps):
    ax.annotate(f"{yk}", (xk, yk), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=9)
ax.set_xlabel("past KV length (context tokens)"); ax.set_ylabel("decode tokens / sec")
ax.set_title("INT4 decode scales gracefully: 31x context -> only 9% slower",
             fontweight="bold")
ax.set_ylim(0, 36)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "decode_scaling.png")); plt.close(fig)

# ---- Fig 4: memory / artifact footprint ----
fig, ax = plt.subplots(figsize=(7, 4.2))
_bars(ax, ["LLM engine\n(Orin, GB)", "tgz\n(GB)", "Load\nVRAM (GB)"],
      [4.85, 5.77, 4.62], [8.25, 7.81, 7.86], "GB",
      "Footprint: INT4 ~42% smaller (fits Orin unified mem comfortably)", fmt="{:.2f}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "footprint.png")); plt.close(fig)

# ---- Fig 5: quantization accuracy drop-off vs FP16 (eval/accuracy/) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
# perplexity under the FP16 model (lower = closer to FP16 distribution)
_bars(a1, ["mean PPL"], [1.359], [1.460], "perplexity (vs FP16=1.237)",
      "Fidelity: +9.8% vs +18.1%", fmt="{:.3f}")
a1.axhline(1.237, color="#16a34a", lw=1.6, ls="--")
a1.annotate("FP16 ref 1.237", (0, 1.237), color="#16a34a", fontsize=9,
            xytext=(0, 4), textcoords="offset points", ha="center")
# token agreement with FP16 greedy path
_bars(a2, ["prefix\nagreement %", "prefix len\n(tok)"], [16.4, 20.1], [5.7, 7.2],
      "% / tokens", "Greedy-path follow (INT4 longer)", fmt="{:.1f}")
fig.suptitle("Quantization drop-off vs FP16 (12 driving/reasoning prompts, greedy): INT4 (AWQ) > INT8 (SQ)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(OUT, "accuracy_dropoff.png")); plt.close(fig)

# ---- Fig 6: task accuracy per subset + overall (Cosmos-Reason1 MC, ground-truth) ----
# Wilson 95% CI (overall n=210). McNemar vs FP16: INT8 p=0.85, INT4 p=0.42 (not significant).
fig, ax = plt.subplots(figsize=(8.4, 4.6))
groups = ["robovqa\n(n=110, easier)", "robofail\n(n=100, harder)", "ALL\n(n=210)"]
fp16 = [88.2, 63.0, 76.2]; int8 = [87.3, 62.0, 75.2]; int4 = [87.3, 59.0, 73.8]
x = np.arange(len(groups)); w = 0.26
for off, vals, c, lab in [(-w, fp16, "#16a34a", "FP16 (ref)"),
                          (0, int8, C8, "INT8 (SmoothQuant)"),
                          (w, int4, C4, "INT4 (AWQ)")]:
    bars = ax.bar(x + off, vals, w, color=c, label=lab)
    for r, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (r.get_x()+r.get_width()/2, v), ha="center", va="bottom",
                    fontsize=8.5, xytext=(0, 2), textcoords="offset points")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("multiple-choice accuracy (%)"); ax.set_ylim(40, 100)
ax.legend(fontsize=9, ncol=3, loc="upper center")
ax.set_title("Task accuracy per subset + overall (Cosmos-Reason1-Benchmark, MC)\n"
             "Quantization: no significant loss on either subset (McNemar p>0.4, overlapping 95% CI)",
             fontweight="bold", fontsize=10.5)
ax.margins(y=0.12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "benchmark_accuracy.png")); plt.close(fig)

# ---- Fig 7: serving metrics + quantization speedup vs FP16 (all on goat, same device) ----
CF = "#16a34a"  # FP16 green
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
# left: TTFT (prefill) & TPOT (decode) grouped bars for FP16/INT8/INT4
metrics = ["TTFT\n(prefill 512)", "TPOT\n(decode /token)"]
fp16 = [298.1, 89.27]; int8 = [159.3, 52.43]; int4 = [304.6, 33.62]
x = np.arange(2); w = 0.26
for off, vals, c, lab in [(-w, fp16, CF, "FP16"), (0, int8, C8, "INT8 (SQ)"), (w, int4, C4, "INT4 (AWQ)")]:
    bars = a1.bar(x + off, vals, w, color=c, label=lab)
    for r, v in zip(bars, vals):
        a1.annotate(f"{v:.0f}", (r.get_x()+r.get_width()/2, v), ha="center", va="bottom",
                    fontsize=8.5, xytext=(0, 2), textcoords="offset points")
a1.set_xticks(x); a1.set_xticklabels(metrics); a1.set_ylabel("latency (ms)")
a1.set_title("TTFT / TPOT vs FP16\ndecode: INT4 2.66× / INT8 1.70× faster than FP16", fontweight="bold", fontsize=10)
a1.legend(fontsize=9, ncol=3, loc="upper center"); a1.margins(y=0.18)
# right: E2E (512-in + 128-out) for the three
labels = ["FP16", "INT8 (SQ)", "INT4 (AWQ)"]; e2e = [11.72, 6.87, 4.61]; cols = [CF, C8, C4]
b = a2.bar(labels, e2e, color=cols, width=0.6)
for r, v, sp in zip(b, e2e, ["1.0× (ref)", "1.7×", "2.5×"]):
    a2.annotate(f"{v:.1f}s\n{sp}", (r.get_x()+r.get_width()/2, v), ha="center", va="bottom",
                fontsize=10, fontweight="bold", xytext=(0, 2), textcoords="offset points")
a2.set_ylabel("E2E latency (s)"); a2.set_ylim(0, 14)
a2.set_title("E2E (512-in + 128-out): quantization → up to 2.5× faster", fontweight="bold", fontsize=10)
a2.margins(y=0.15)
fig.suptitle("Serving metrics & quantization speedup vs FP16  (orin-goat 64GB, MAXN, batch=1; INT4/INT8 within 3% of orin-dog)", fontsize=10.5)
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(os.path.join(OUT, "serving_metrics.png")); plt.close(fig)

print("wrote:", ", ".join(sorted(os.listdir(OUT))))

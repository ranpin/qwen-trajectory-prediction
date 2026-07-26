#!/usr/bin/env python3
"""Roofline analysis of Cosmos-Reason2-8B prefill/decode on Jetson AGX Orin.

Explains the three measured anomalies mechanistically:
  1. decode is DRAM-bound  -> all precisions sit near the achievable-bandwidth
     ceiling, and the INT4 speedup is capped by bytes/token, not by FLOPs;
  2. prefill is compute-bound -> INT4 (W4A16) lands on the *FP16* ceiling
     because TensorRT's weight-only path dequantizes before the MMA, so it
     cannot beat FP16;
  3. INT8 (W8A8) lands on the 2x-higher INT8 tensor ceiling -> ~1.87x prefill.

MODEL SIZE — derived analytically from the measured config and cross-checked
against the measured engine size (a self-consistency test, not an assumption):
  per layer: attn q/k/v/o = 4096*4096 + 2*(4096*1024) + 4096*4096 = 41.94 M
             MLP gate/up/down = 3 * 4096 * 12288                 = 150.99 M
  36 layers * 192.94 M = 6.946 B ; + lm_head 151936*4096 = 0.622 B
  => 7.575 B matmul params ; 7.575 B * 2 B/param = 15.15 GB == measured FP16
     engine size (docs/edge_deploy_status.md). Adding the input embedding table
     (0.622 B, shipped as a separate fp16 file) and the vision encoder (~0.58 B)
     reproduces the 8.77 B total that NVIDIA publishes for the checkpoint.

HARDWARE — NVIDIA primary sources (see docs/METHODOLOGY.md for URLs):
  AGX Orin 64GB / Dev-Kit-class silicon, sm_87, MAXN, 1301 MHz, 2048 CUDA
  cores / 64 Tensor Cores:
    peak DRAM bandwidth 204.8 GB/s (256-bit LPDDR5 @ 6400 MT/s)
    dense GPU tensor peak: FP16 43 TFLOP/s, INT8 85 TOP/s
  The headline "275 TOPS" is sparse INT8 *including the 2x DLAs* (170 GPU
  sparse + 105 DLA sparse); a GPU-only dense roofline must use 85 TOPS.
  INT4: sm_87 has .s4 mma in the ISA, but TensorRT weight-only-quantization
  dequantizes INT4 weights to high precision before the dot product, so the
  W4A16 compute ceiling is the FP16 one.

Run:  .venv/bin/python alpamayo-edge/eval/roofline.py
Out:  alpamayo-edge/docs/figures/roofline.png  + a verification table on stdout
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply, ANNOT, C4, C8, CF, WIDTH_FULL)

apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- hardware ceilings (AGX Orin, sm_87, MAXN) ---------------------------
BW_PEAK = 204.8e9          # B/s, theoretical
BW_ACH = 150e9             # B/s, achievable (~73% of peak; see METHODOLOGY)
PEAK_FP16 = 43e12          # FLOP/s, dense GPU tensor cores
PEAK_INT8 = 85e12          # OP/s,   dense GPU tensor cores

# ---- model ----------------------------------------------------------------
W_MATMUL = 7.575e9         # matmul params (validated against FP16 engine size)
PREFILL_TOKENS = 512
FLOPS_DECODE = 2 * W_MATMUL                    # per generated token
FLOPS_PREFILL = 2 * W_MATMUL * PREFILL_TOKENS  # 512-token prompt

# ---- measured (orin-goat, MAXN, batch=1; eval/perf/fp16_speedup_goat.json) --
# engine bytes streamed per decode step, and measured latencies
CASES = [
    # name,      engine_bytes, ttft_s,  tpot_s,  ceiling,    color
    ("FP16",       15.15e9,     0.2981, 0.08927, PEAK_FP16, CF),
    ("INT8 (SQ)",   8.30e9,     0.1593, 0.05243, PEAK_INT8, C8),
    ("INT4 (AWQ)",  4.80e9,     0.3046, 0.03362, PEAK_FP16, C4),
]

print(f"{'':12} {'AI dec':>7} {'AI pre':>8} | {'dec BW':>9} {'%peak':>6} | "
      f"{'pre TF/s':>9} {'%ceil':>6}")
rows = []
for name, byts, ttft, tpot, ceil, col in CASES:
    ai_dec = FLOPS_DECODE / byts                 # FLOP/byte
    ai_pre = FLOPS_PREFILL / byts
    bw_dec = byts / tpot                         # achieved B/s
    perf_dec = FLOPS_DECODE / tpot               # achieved FLOP/s
    perf_pre = FLOPS_PREFILL / ttft
    rows.append((name, ai_dec, ai_pre, perf_dec, perf_pre, ceil, col))
    print(f"{name:12} {ai_dec:7.2f} {ai_pre:8.0f} | {bw_dec/1e9:8.1f}G "
          f"{100*bw_dec/BW_PEAK:5.1f}% | {perf_pre/1e12:8.1f} "
          f"{100*perf_pre/ceil:5.1f}%")

RIDGE_FP16 = PEAK_FP16 / BW_PEAK
RIDGE_INT8 = PEAK_INT8 / BW_PEAK
print(f"\nridge point: FP16 {RIDGE_FP16:.0f} FLOP/byte, INT8 {RIDGE_INT8:.0f} OP/byte")
print(f"decode AI is {RIDGE_FP16/rows[0][1]:.0f}-{RIDGE_FP16/rows[2][1]:.0f}x "
      f"below the ridge -> hard memory-bound; prefill AI is above it -> compute-bound")

# ---- figure ---------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 4.25))

ai = np.logspace(-1, 4, 400)
for ax in (a1, a2):
    # memory roofs (slanted) and compute roofs (flat)
    ax.plot(ai, np.minimum(BW_PEAK * ai, PEAK_INT8) / 1e12, "-", color="#555555",
            lw=1.2, label="INT8 roof (85 TOP/s, 204.8 GB/s)")
    ax.plot(ai, np.minimum(BW_PEAK * ai, PEAK_FP16) / 1e12, "--", color="#555555",
            lw=1.0, label="FP16 roof (43 TFLOP/s, 204.8 GB/s)")
    ax.plot(ai, np.minimum(BW_ACH * ai, PEAK_FP16) / 1e12, ":", color="#999999",
            lw=1.0, label="FP16 roof at achievable BW (150 GB/s)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP or OP per byte)")
    ax.grid(True, which="both", axis="both")

a1.set_ylabel("attainable throughput (TFLOP/s or TOP/s)")

# left panel: decode operating points
for name, ai_dec, ai_pre, perf_dec, perf_pre, ceil, col in rows:
    a1.plot(ai_dec, perf_dec / 1e12, "o", color=col, ms=6, zorder=5)
    a1.annotate(name, (ai_dec, perf_dec / 1e12), fontsize=ANNOT,
                xytext=(7, 0), textcoords="offset points",
                va="center", ha="left", color=col)
a1.set_title("Decode (per token, batch = 1): memory-bound")
a1.set_xlim(0.3, 1e4); a1.set_ylim(1e-2, 2e2)

# right panel: prefill operating points
for name, ai_dec, ai_pre, perf_dec, perf_pre, ceil, col in rows:
    a2.plot(ai_pre, perf_pre / 1e12, "s", color=col, ms=6, zorder=5)
    a2.annotate(name, (ai_pre, perf_pre / 1e12), fontsize=ANNOT,
                xytext=(7, 0), textcoords="offset points",
                va="center", ha="left", color=col)
a2.set_title("Prefill (512-token prompt): compute-bound")
a2.set_xlim(0.3, 1e4); a2.set_ylim(1e-2, 2e2)

handles, labels = a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
fig.suptitle("Roofline on Jetson AGX Orin (sm_87, MAXN, batch = 1)")
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(os.path.join(OUT, "roofline.png")); plt.close(fig)
print("\nwrote", os.path.join(OUT, "roofline.png"))

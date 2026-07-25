#!/usr/bin/env python3
"""Kernel-level decode breakdown from Nsight Systems (INT4, orin-dog).

Source data: eval/profile/int4_decode_kern_sum.csv, produced on the device by
    nsys profile --trace cuda --sample none --cuda-graph-trace=node \
        llm_bench --engineDir engines/int4/llm --mode decode --pastKVLen 512 \
                  --iterations 20 --warmup 3
    nsys stats --report cuda_gpu_kern_sum --format csv
`--cuda-graph-trace=node` is essential: without it nsys collapses the 20 timed
iterations (which run inside a captured CUDA graph) into one range and only the
3 slow warmup iterations get per-kernel attribution.

Byte counts are derived from the measured config (36 layers, hidden 4096, FFN
12288, GQA 32/8, vocab 151936, AWQ group 128) and close to within +0.12% of the
measured 4.847 GB engine file -> the decomposition is verified, not assumed.
Effective bandwidth = derived bytes / measured kernel time, because Nsight
Compute hardware counters need root and sudo is password-locked on this device.

Run:  .venv/bin/python alpamayo-edge/eval/profile_viz.py
Out:  alpamayo-edge/docs/figures/kernel_breakdown.png
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import apply, bar_kw, ANNOT, C4, C8, CF, RULE

apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")

ITERS = 24                 # 3 warmup + 1 capture + 20 timed, all traced as nodes
TPOT_MS = 33.1313          # measured E2E decode latency per token
PEAK_BW = 204.8e9

# ---- measured kernel times (ns, summed over ITERS) ------------------------
KERN = {
    "W4A16 GEMV\n(7x per layer)": 585395318,
    "lm_head GEMM\n(fp16, N=151936)": 178950924,
    "attention\n(kernel_mha)": 21814462,
    "fused elementwise\n(RMSNorm/RoPE/SwiGLU)": 6112496 + 5453216 + 3984287
                                                + 3833968 + 3560959 + 2894719,
}
COLORS = [C4, RULE, C8, CF]

# ---- derived bytes streamed per token -------------------------------------
V, H, L, FFN, KVH, HD, GRP = 151936, 4096, 36, 12288, 8, 128, 128
quant_w = L * (2 * H * H + 2 * H * KVH * HD + 3 * H * FFN)
BYTES = {
    "INT4 weights": quant_w * 0.5,
    "AWQ scales+zeros": quant_w / GRP * 2.5,
    "lm_head fp16\n(NOT quantized)": H * V * 2,
}
BCOLORS = [C4, C8, RULE]
MEASURED_ENGINE = 4.847e9

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.25))

# left: per-token time attribution
names = list(KERN)
ms = np.array([KERN[k] / ITERS / 1e6 for k in names])
y = np.arange(len(names))[::-1]
bars = a1.barh(y, ms, height=0.6, color=COLORS, **bar_kw())
for r, v in zip(bars, ms):
    a1.annotate(f"{v:.2f} ms  ({100*v/ms.sum():.1f}%)",
                (v, r.get_y() + r.get_height() / 2), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=ANNOT)
a1.set_yticks(y); a1.set_yticklabels(names)
a1.set_xlabel("time per generated token (ms)")
a1.set_xlim(0, ms.max() * 1.45)
a1.set_title(f"Decode time by kernel (sum {ms.sum():.1f} ms ~ measured TPOT {TPOT_MS:.1f} ms)")
a1.grid(True, axis="x"); a1.grid(False, axis="y")

# right: engine bytes streamed per token
bn = list(BYTES)
gb = np.array([BYTES[k] / 1e9 for k in bn])
y2 = np.arange(len(bn))[::-1]
bars2 = a2.barh(y2, gb, height=0.6, color=BCOLORS, **bar_kw())
for r, v in zip(bars2, gb):
    a2.annotate(f"{v:.3f} GB  ({100*v/gb.sum():.1f}%)",
                (v, r.get_y() + r.get_height() / 2), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=ANNOT)
a2.set_yticks(y2); a2.set_yticklabels(bn)
a2.set_xlabel("bytes streamed per token (GB)")
a2.set_xlim(0, gb.max() * 1.45)
a2.set_title(f"Engine bytes by role (sum {gb.sum():.3f} GB = measured "
             f"{MEASURED_ENGINE/1e9:.3f} GB +{100*(gb.sum()*1e9-MEASURED_ENGINE)/MEASURED_ENGINE:.2f}%)")
a2.grid(True, axis="x"); a2.grid(False, axis="y")

fig.suptitle("Kernel-level decode breakdown (INT4 AWQ, orin-dog, MAXN, batch = 1, pastKV 512)")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "kernel_breakdown.png")); plt.close(fig)

# ---- effective bandwidth + headroom --------------------------------------
gemv_s = KERN["W4A16 GEMV\n(7x per layer)"] / ITERS / 1e9
lm_s = KERN["lm_head GEMM\n(fp16, N=151936)"] / ITERS / 1e9
gemv_b = BYTES["INT4 weights"] + BYTES["AWQ scales+zeros"]
lm_b = BYTES["lm_head fp16\n(NOT quantized)"]
tot_b = gb.sum() * 1e9
print(f"gemv    {gemv_b/gemv_s/1e9:6.1f} GB/s ({100*gemv_b/gemv_s/PEAK_BW:4.1f}% of peak)")
print(f"lm_head {lm_b/lm_s/1e9:6.1f} GB/s ({100*lm_b/lm_s/PEAK_BW:4.1f}% of peak)")
print(f"whole   {tot_b/(TPOT_MS/1e3)/1e9:6.1f} GB/s "
      f"({100*tot_b/(TPOT_MS/1e3)/PEAK_BW:4.1f}% of peak)")
lm_int4 = H * V * 0.5 + H * V / GRP * 2.5
new_b = gemv_b + lm_int4
print(f"\nheadroom A - quantize lm_head to INT4: {tot_b/1e9:.3f} -> {new_b/1e9:.3f} GB "
      f"=> TPOT ~{TPOT_MS*new_b/tot_b:.1f} ms ({1000/(TPOT_MS*new_b/tot_b):.1f} tok/s, "
      f"+{100*(tot_b/new_b-1):.0f}%)")
print(f"headroom B - lift GEMV 72.2% -> 85% of peak: TPOT ~"
      f"{(gemv_b/(0.85*PEAK_BW) + lm_s + (ms.sum()-gemv_s*1e3-lm_s*1e3)/1e3)*1e3:.1f} ms")
print("\nwrote", os.path.join(OUT, "kernel_breakdown.png"))

#!/usr/bin/env python3
"""Vision-tower cost and multi-camera scaling (INT4 LLM + fp16 vision tower).

Data: eval/vision/RESULTS.json, measured on orin-dog with
  llm_inference --dumpProfile --warmup 1  (TRT-reported GPU time per stage)

Run:  .venv/bin/python alpamayo-edge/eval/vision/vision_viz.py
Out:  alpamayo-edge/docs/figures/vision_tower.png
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from paper_style import apply, bar_kw, ANNOT, C4, C8, CF, RULE, WIDTH_FULL

apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "docs", "figures")
D = json.load(open(os.path.join(HERE, "RESULTS.json")))
MAX_INPUT_LEN = D["constraints"]["llm_maxInputLen"]
TOK_PER_FRAME = D["per_frame_image_tokens"]

f = np.array([r["frames"] for r in D["runs"]], float)
vis = np.array([r["vision_ms"] for r in D["runs"]])
pre = np.array([r["prefill_ms"] for r in D["runs"]])
ttft = vis + pre

fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 4.25))

# ---- left: TTFT split into vision encode + LLM prefill ----
x = np.arange(len(f))
b1 = a1.bar(x, vis, 0.55, label="vision tower (fp16)", color=C8, **bar_kw())
b2 = a1.bar(x, pre, 0.55, bottom=vis, label="LLM prefill (INT4)", color=C4, **bar_kw())
for i in range(len(f)):
    a1.annotate(f"{ttft[i]:.0f} ms", (x[i], ttft[i]), ha="center", va="bottom",
                fontsize=ANNOT, xytext=(0, 2), textcoords="offset points")
    a1.annotate(f"{100*vis[i]/ttft[i]:.0f}%", (x[i], vis[i] / 2), ha="center",
                va="center", fontsize=ANNOT, color="#1a1a2e")
a1.set_xticks(x); a1.set_xticklabels([f"{int(n)}" for n in f])
a1.set_xlabel("input frames")
a1.set_ylabel("time to first token (ms)")
a1.set_title("TTFT split: vision tower is 19-25% of it")
a1.set_ylim(0, ttft.max() * 1.18)

# ---- right: vision cost scaling + AV camera-count extrapolation ----
slope, icept = np.polyfit(f, vis, 1)
xs = np.linspace(0.5, 9.2, 100)
a2.plot(xs, icept + slope * xs, "-", color="#888888", lw=1.0,
        label=f"fit: {icept:.1f} + {slope:.1f} ms x frames")
a2.plot(f, vis, "o", color=C8, ms=6, label="measured")
for xi, yi in zip(f, vis):
    a2.annotate(f"{yi:.0f}", (xi, yi), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=ANNOT)
a2.plot([8], [icept + slope * 8], "s", color=RULE, ms=6,
        label="8 cameras (extrapolated)")
a2.annotate(f"{icept+slope*8:.0f} ms", (8, icept + slope * 8), xytext=(0, 7),
            textcoords="offset points", ha="center", fontsize=ANNOT, color=RULE)

# the engine's maxInputLen is a hard wall on how many frames can be fed at all
max_frames = (MAX_INPUT_LEN - 77) / TOK_PER_FRAME
a2.axvline(max_frames, color=RULE, ls="--", lw=1.0)
a2.annotate(f"engine maxInputLen={MAX_INPUT_LEN}\n=> {int(max_frames)} frames max",
            (max_frames, vis.max() * 0.35), xytext=(-6, 0),
            textcoords="offset points", ha="right", fontsize=ANNOT, color=RULE)
a2.set_xlabel("input frames (= cameras, 1 frame each)")
a2.set_ylabel("vision tower GPU time (ms)")
a2.set_title(f"Vision tower scales linearly: {slope:.1f} ms/frame")
a2.set_xlim(0, 9.4); a2.set_ylim(0, 235)
a2.legend(loc="upper left")

h1, l1 = a1.get_legend_handles_labels()
fig.legend(h1, l1, loc="lower center", ncol=2, frameon=False)
fig.suptitle(f"Vision tower cost, {TOK_PER_FRAME} tokens/frame @<=448 px "
             "(orin-dog, INT4 LLM + fp16 vision tower, batch = 1)")
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(os.path.join(OUT, "vision_tower.png")); plt.close(fig)

print(f"vision fit: {icept:.1f} + {slope:.2f} ms/frame (R2="
      f"{np.corrcoef(f, vis)[0,1]**2:.4f})")
for i in range(len(f)):
    print(f"  {int(f[i])} frames: vision {vis[i]:6.2f} ms  prefill {pre[i]:6.2f} ms  "
          f"TTFT {ttft[i]:6.1f} ms  vision share {100*vis[i]/ttft[i]:.1f}%")
print(f"  8 frames (AV): vision ~{icept+slope*8:.0f} ms, input tokens "
      f"{int(8*TOK_PER_FRAME+77)} vs maxInputLen {MAX_INPUT_LEN} -> fits, 9 does not")
print("wrote", os.path.join(OUT, "vision_tower.png"))

#!/usr/bin/env python3
"""One figure that answers "what goes in and what comes out" with real data.

Everything shown is verbatim from a real run of the deployed INT4 engine on orin-dog:
frames are the actual JPEGs fed to the vision tower, the prompt is the actual text, and
both outputs are the actual generations (freeform_out.json, and the n=210 benchmark dump).

Run:  .venv/bin/python alpamayo-edge/eval/io_example/io_viz.py
Out:  alpamayo-edge/docs/figures/io_example.png
"""
import json
import os
import sys
import textwrap

import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from paper_style import apply, ANNOT, BASE, C4, C8, CF, INK

apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "docs", "figures")

FRAMES = [os.path.join(HERE, f"robovqa_0_{i}.jpg") for i in range(6)]
GEN = json.load(open(os.path.join(HERE, "freeform_out.json")))["responses"][0]["output_text"]
PROMPT = ("You are shown 6 frames sampled in order from a video of a robot performing a task. "
          "Describe what happens across the frames, then judge whether the action looks safe "
          "and whether the task succeeded. Answer in 2-3 sentences.")

fig = plt.figure(figsize=(10.4, 4.75))
gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 0.80, 1.10], hspace=0.45, wspace=0.06)

# ---- row 1: the actual image input ----
for i, f in enumerate(FRAMES):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(mpimg.imread(f))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color("#888888")
    ax.set_title(f"frame {i+1}", fontsize=ANNOT, pad=2)
fig.text(0.5, 0.995, "INPUT 1 of 2 — vision:  6 video frames, resized to <=448 px  "
                     "->  vision tower  ->  112 image tokens per frame = 672 tokens",
         ha="center", fontsize=BASE, color=INK)

# ---- row 2: the actual text input ----
ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.grid(False)
ax.text(0, 1.0, "INPUT 2 of 2 — text:  the question, verbatim (87 tokens)",
        fontsize=BASE, va="top", color=INK)
ax.text(0, 0.66, "\n".join(textwrap.wrap(PROMPT, 118)), fontsize=ANNOT, va="top",
        family="monospace", color="#333333",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f2f6ff", edgecolor=C4, linewidth=0.8))
ax.text(0, -0.05, "total prompt = 672 image + 87 text = 759 tokens  ->  prefill",
        fontsize=ANNOT, va="bottom", color="#666666")

# ---- row 3: the actual output ----
ax = fig.add_subplot(gs[2, :]); ax.axis("off"); ax.grid(False)
ax.text(0, 1.0, "OUTPUT — generated text, verbatim from the INT4 engine on orin-dog",
        fontsize=BASE, va="top", color=INK)
ax.text(0, 0.80, "\n".join(textwrap.wrap(GEN, 118)), fontsize=ANNOT, va="top",
        family="monospace", color="#333333",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1fbf4", edgecolor=CF, linewidth=0.8))
ax.text(0, -0.30, "Same engine, benchmark mode: the question ends with "
                  "\"Answer with ONLY the single letter\" -> output is one token, e.g. \"A\", "
                  "which is what the n=210 accuracy is scored on.",
        fontsize=ANNOT, va="bottom", color="#666666")

fig.suptitle("", y=0.99)
fig.savefig(os.path.join(OUT, "io_example.png"), bbox_inches="tight")
plt.close(fig)
print("generation length:", len(GEN), "chars")
print("wrote", os.path.join(OUT, "io_example.png"))

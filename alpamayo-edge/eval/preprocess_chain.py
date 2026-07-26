#!/usr/bin/env python3
"""From a benchmark video to image tokens: every step with the real numbers.

Written because the earlier figures said "N frames, <=448 px, 112 tokens per
frame" without saying which model does the encoding, what preprocessing runs,
what size actually reaches the network, or that 112 is specific to one frame
shape. All numbers here are measured, not assumed:

  frame shapes    PIL over the 1260 jpgs actually used on orin-dog
                  (~/bench/frames): 1020x 448x252 + 240x 448x358
  resize rule     engines/int4/visual/preprocessor_config.json ->
                  Qwen2VLImageProcessor, patch 16, merge 2, bicubic,
                  pixel budget [100352, 2097152] -> side rounded to a
                  multiple of patch*merge = 32
  token counts    llm_inference --dumpProfile on the device:
                  6x 448x252 -> "Total Image Tokens: 672"
                  6x 448x358 -> "Total Image Tokens: 924"  (154 per frame)
  prompt lengths  same runs, "Computed Tokens": 759 and 951; the worst item of
                  the n=210 benchmark (request_idx 94) measures 1015 against
                  the engine's maxInputLen = 1024 -> 9 tokens of headroom
Raw values: eval/vision/RESULTS.json (preprocessing, runs_448x358_2026_07_26,
max_input_len_headroom_2026_07_26).

Run:  .venv/bin/python alpamayo-edge/eval/preprocess_chain.py
Out:  alpamayo-edge/docs/figures/preprocess_chain.png
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import apply, bar_kw, ANNOT, BASE, C4, C8, INK, RULE, TITLE

apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")

WIDE, TALL = C4, C8          # 448x252 (16:9) and 448x358 colours
SMALL = ANNOT - 0.5

fig = plt.figure(figsize=(10.4, 6.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.42)

# ---------------- (a) the chain -------------------------------------------
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.grid(False)

STAGES = [
    ("① benchmark 视频", [("robovqa 110 段", INK), ("robofail 100 段", INK),
                          ("（仓库自带 mp4）", "#64748b")]),
    ("② 我们做的采样", [("均匀取 6 帧", INK), ("长边缩到 448 px", INK),
                        ("共 1260 张 jpg", "#64748b")]),
    ("③ 实测帧尺寸", [("448×252（170 段）", WIDE), ("448×358（40 段）", TALL),
                      ("保持原长宽比", "#64748b")]),
    ("④ processor 缩放", [("bicubic → 32 的整数倍", INK),
                          ("448×256", WIDE), ("448×352", TALL)]),
    ("⑤ 归一化 + 切 patch", [("/255 → 均值=标准差=0.5", INK),
                             ("16×16：28×16=448 块", WIDE),
                             ("16×16：28×22=616 块", TALL)]),
    ("⑥ ViT + 2×2 merge", [("每 token 覆盖 32×32 px", INK),
                           ("112 token / 帧", WIDE), ("154 token / 帧", TALL)]),
]
W, GAP = 15.0, 2.0
for i, (head, lines) in enumerate(STAGES):
    x = i * (W + GAP)
    ax.add_patch(FancyBboxPatch((x, 8.0), W, 74.0,
                 boxstyle="round,pad=0,rounding_size=1.4", mutation_aspect=0.30,
                 linewidth=0.9, edgecolor="#94a3b8", facecolor="#f8fafc"))
    ax.text(x + W / 2, 74.0, head, ha="center", va="center", fontsize=ANNOT,
            fontweight="bold", color="#334155")
    for j, (txt, col) in enumerate(lines):
        ax.text(x + W / 2, 56.0 - j * 17.0, txt, ha="center", va="center",
                fontsize=SMALL, color=col)
    if i:
        ax.annotate("", xy=(x - 0.4, 45.0), xytext=(x - GAP + 0.4, 45.0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#94a3b8",
                                    shrinkA=0, shrinkB=0))
ax.text(50, 91.0,
        "真正进入网络的不是 448×252，而是缩到 32 的整数倍后的 448×256（每帧 112 token）"
        "；40 段 5:4 视频是 448×352（每帧 154 token）",
        ha="center", va="center", fontsize=ANNOT, color=INK)
ax.text(50, 0.5, "视觉编码器 = Qwen3-VL 的 ViT（qwen3_vl_vision，27 层，fp16 未量化）"
                 "；预处理全部在主机侧 CPU 完成，不在任何 TRT 引擎里",
        ha="center", va="center", fontsize=SMALL, color="#64748b")

# ---------------- (b) prompt-length budget --------------------------------
ax2 = fig.add_subplot(gs[1])
CASES = [
    ("6 帧 448×252 + 自由问答提问\n（§2.1 的逐字实例）", 672, 87, WIDE),
    ("6 帧 448×358 + 极短提问\n（本次补测）", 924, 27, TALL),
    ("6 帧 448×358 + 最长的基准题\n（n=210 里的第 94 题，最坏情况）", 924, 91, TALL),
]
y = np.arange(len(CASES))[::-1]
for (lab, img, txt, col), yy in zip(CASES, y):
    ax2.barh(yy, img, color=col, height=0.52, **bar_kw())
    ax2.barh(yy, txt, left=img, color="#cbd5e1", height=0.52, **bar_kw())
    ax2.text(1150, yy, f"{img} + {txt} = {img + txt}", va="center",
             fontsize=ANNOT, color=INK)
ax2.axvline(1024, color=RULE, ls="--", lw=1.1)
ax2.annotate("引擎硬上限\nmaxInputLen\n= 1024", (1024, y[0] + 0.30),
             xytext=(5, 0), textcoords="offset points", ha="left", va="center",
             fontsize=ANNOT, color=RULE)
ax2.set_yticks(y)
ax2.set_yticklabels([c[0] for c in CASES], fontsize=SMALL)
ax2.set_xlim(0, 1420)
ax2.set_ylim(-0.6, len(CASES) - 0.25)
ax2.set_xlabel("一次请求的 prompt 长度（token；深色=图像 token，浅灰=问题与对话模板）")
ax2.grid(True, axis="x")
ax2.set_axisbelow(True)
ax2.set_title("Prompt length against the engine's hard input limit "
              "(measured on orin-dog, INT4)")

fig.suptitle("From a benchmark video to image tokens: measured preprocessing "
             "chain and the resulting prompt budget", y=0.995, fontsize=TITLE)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(os.path.join(OUT, "preprocess_chain.png"), bbox_inches="tight",
            facecolor="white")
plt.close(fig)

print("448x252 -> 448x256 :", (448 // 32), "x", (256 // 32), "=", (448 // 32) * (256 // 32), "tokens")
print("448x358 -> 448x352 :", (448 // 32), "x", (352 // 32), "=", (448 // 32) * (352 // 32), "tokens")
print("wrote", os.path.join(OUT, "preprocess_chain.png"))

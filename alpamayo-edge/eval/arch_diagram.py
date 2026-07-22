#!/usr/bin/env python3
"""Render the end-to-end pipeline / architecture diagram (真正绘制，非 ASCII).

Out: docs/figures/pipeline.png  (+ copied to site/figures/ by the caller)
Run: .venv/bin/python alpamayo-edge/eval/arch_diagram.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# color roles
SRC = "#e9ecef"      # source weights (gray)
CLOUD = "#dbeafe"    # cloud step (blue)
CLOUD_E = "#3b82f6"
ART = "#fef3c7"      # artifact (amber-ish)
EDGE = "#dcfce7"     # edge step (green)
EDGE_E = "#22c55e"
EVAL = "#ede9fe"     # eval (purple)
EVAL_E = "#8b5cf6"

fig, ax = plt.subplots(figsize=(8.2, 9.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


def box(y, text, face, edge="#94a3b8", x=5.0, w=8.6, h=1.0, fs=12.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.04,rounding_size=0.12",
                 linewidth=1.6, edgecolor=edge, facecolor=face))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1a1a2e", linespacing=1.5)


def arrow(y1, y2, label=None, x=5.0):
    ax.annotate("", xy=(x, y2 + 0.02), xytext=(x, y1 - 0.02),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#475569"))
    if label:
        ax.text(x + 0.15, (y1 + y2) / 2, label, ha="left", va="center",
                fontsize=10.5, color="#475569", style="italic")


box(9.2, "nvidia/Cosmos-Reason2-8B\nHuggingFace 门控 · fp16 预训练权重（我们只量化、不训练）",
    SRC, x=5, w=8.6, h=1.0, fs=12.5)
arrow(8.7, 8.05, "下载")
box(7.55, "云端 Kaggle T4×2  —  NVIDIA ModelOpt 训练后量化(PTQ)\nAWQ (W4A16)  /  SmoothQuant (W8A8)   ·   校准集: cnn_dailymail 512 篇",
    CLOUD, CLOUD_E, w=9.0, h=1.15, fs=12.5)
arrow(6.97, 6.3, "导出 ONNX")
box(5.85, "边缘 ONNX 产物   (LLM + fp16 视觉塔)", ART, "#d97706", w=6.6, h=0.85)
arrow(5.42, 4.75, "scp  (Orin 无外网：云 → 本地 → Orin)")
box(4.3, "Jetson AGX Orin (sm_87)  —  TensorRT-Edge-LLM v0.9.0  ·  llm_build",
    EDGE, EDGE_E, w=9.0, h=0.95, fs=12.5)
arrow(3.82, 3.2, "建引擎")
box(2.75, "TensorRT 引擎    INT4 / INT8 / FP16   (设备专用)", ART, "#d97706", w=7.2, h=0.85)
arrow(2.32, 1.7)
box(1.25, "边缘推理 llm_inference / llm_bench   +   tegrastats 采功耗", EDGE, EDGE_E, w=8.6, h=0.85)
arrow(0.82, 0.2)
box(-0.35, "评测   ·  性能：延迟 / 吞吐 / 显存 / 功耗 / 能效\n·  精度：vs FP16 参考 perplexity + token 一致率",
    EVAL, EVAL_E, w=9.0, h=1.05, fs=12.5)
ax.set_ylim(-1.0, 10)

ax.text(5, 9.86, "Alpamayo-Edge 端到端流程：云端量化 → 边缘部署 → 实测评测",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color="#1a1a2e")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "pipeline.png"), dpi=150, bbox_inches="tight",
            facecolor="white")
print("wrote", os.path.join(OUT, "pipeline.png"))

#!/usr/bin/env python3
"""Model structure of Cosmos-Reason2-8B with every optimization's landing point.

Why this figure exists: the results sections quote per-module shares (72.1% of
decode time in the W4A16 GEMV, 25.6% of decode bytes in an unquantized lm_head,
19-25% of TTFT in the vision encoder) but the reader had no map of the model to
attach them to. This draws the actual measured architecture on the left and, on
the right, what we did to each module and what it measured.

Every number here is either read from the measured config or computed from it and
already cross-validated elsewhere in this repo:
  * shapes: 36 layers, hidden 4096, GQA 32/8 heads, headDim 128, FFN 12288,
    vocab 151936  (edge_int4/.../llm/config.json + Orin LLMEngineConfig log)
  * parameter split and INT4 decode bytes: same formula as eval/profile_viz.py,
    which agrees with the measured 4.847 GB engine file to +0.13%
  * kernel time shares: eval/profile/int4_decode_kern_sum.csv (Nsight Systems)
  * vision encoder: eval/vision/RESULTS.json

Run:  .venv/bin/python alpamayo-edge/eval/model_arch.py
Out:  alpamayo-edge/docs/figures/model_arch.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import apply, ANNOT, BASE, INK, TITLE

apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- measured shapes ------------------------------------------------------
L, H, FFN, KVH, HD, V, GRP = 36, 4096, 12288, 8, 128, 151936, 128
attn_l = 2 * H * H + 2 * H * KVH * HD          # q,o + k,v (GQA)
mlp_l = 3 * H * FFN                            # gate, up, down
quant_w = L * (attn_l + mlp_l)                 # everything AWQ/SQ touches
lm_head = H * V
matmul = quant_w + lm_head
BY_INT4 = quant_w * 0.5 + quant_w / GRP * 2.5 + lm_head * 2   # bytes / token

P_ATTN = 100 * L * attn_l / matmul
P_MLP = 100 * L * mlp_l / matmul
P_LMH = 100 * lm_head / matmul
B_LMH = 100 * lm_head * 2 / BY_INT4

print(f"matmul params {matmul/1e9:.3f} B  = decoder attn {P_ATTN:.1f}% + "
      f"MLP {P_MLP:.1f}% + lm_head {P_LMH:.1f}%")
print(f"INT4 decode bytes/token {BY_INT4/1e9:.3f} GB  -> lm_head {B_LMH:.1f}%")

# ---- palette by "what we did to it" --------------------------------------
Q_F, Q_E = "#dbeafe", "#2563eb"     # quantized by us (INT4/INT8)
F_F, F_E = "#fef3c7", "#d97706"     # still fp16 -> the identified headroom
N_F, N_E = "#eef2f7", "#94a3b8"     # no weights / not in the LLM engine
DS_E = "#7c3aed"                    # deepstack side path
IO_F, IO_E = "#f1f5f9", "#64748b"   # I/O
INNER = "#ffffff"

fig, ax = plt.subplots(figsize=(10.4, 7.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.grid(False)

LX, LW = 2.0, 52.0                  # architecture lane
RX = 56.5                           # annotation lane


def box(x, y, w, h, text, face, edge, fs=BASE, bold=False, ls="-", z=2,
        va="center", ha="center", lsp=1.45):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=1.0", mutation_aspect=0.55,
                 linewidth=0.9, edgecolor=edge, facecolor=face, linestyle=ls,
                 zorder=z))
    tx = x + w / 2 if ha == "center" else x + 1.6
    ty = y + h / 2 if va == "center" else y + h - 1.0
    ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", zorder=z + 1,
            linespacing=lsp)


def down(x, y1, y2, label=None, lw=1.1, color="#475569", style="-|>"):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text(x + 0.9, (y1 + y2) / 2, label, ha="left", va="center",
                fontsize=ANNOT, color="#475569")


def tag(x, y, letter, color):
    """Letter tag on a module; the matching note in the right lane repeats it.

    Tags rather than leader lines: the annotated modules sit in two columns, so
    any leader line to the right lane would have to cross a neighbouring box.
    """
    ax.text(x, y, letter, ha="center", va="center", fontsize=ANNOT - 0.5,
            color=color, fontweight="bold", zorder=7,
            bbox=dict(boxstyle="circle,pad=0.32", facecolor="white",
                      edgecolor=color, linewidth=0.9))


def note(y, letter, title, body, color):
    """Right-lane annotation keyed to a module tag."""
    ax.text(RX + 1.0, y + 1.0, letter, ha="center", va="center",
            fontsize=ANNOT - 0.5, color=color, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.32", facecolor="white",
                      edgecolor=color, linewidth=0.9))
    ax.text(RX + 3.4, y + 0.4, title, ha="left", va="bottom",
            fontsize=ANNOT, color=color, fontweight="bold")
    ax.text(RX + 3.4, y - 0.5, body, ha="left", va="top", fontsize=ANNOT,
            color="#333333", linespacing=1.5)


# ================= input =================
box(LX + 1.0, 92.0, 24.0, 6.4,
    "输入 ①　N 帧图像\n长边 448 px → 32 的整数倍（见图 2）", IO_F, IO_E, fs=ANNOT + 0.5)
box(LX + 27.5, 92.0, 24.0, 6.4,
    "输入 ②　文字问题\n（自然语言 prompt）", IO_F, IO_E)

# ================= vision encoder / embedding =================
down(LX + 13.0, 92.0, 85.4)
down(LX + 39.5, 92.0, 85.4)
box(LX + 1.0, 78.0, 24.0, 7.4,
    "视觉编码器 ViT　qwen3_vl_vision\n27 层 · d=1152 · ≈0.58 B（fp16 未量化）\n"
    "→ 每帧 112 token（448×256）",
    F_F, F_E, fs=ANNOT + 0.5)
box(LX + 27.5, 78.0, 24.0, 7.4,
    "Qwen3 分词器 + 词嵌入表 151936×4096\n运行时 kernel 查表（不在 TRT 图内）\n"
    "独立权重 embedding.safetensors 1.245 GB",
    N_F, N_E, fs=ANNOT + 0.5)

tag(LX + 23.2, 83.8, "A", F_E)
note(84.2, "A", "§1.7　视觉编码器剖析（唯一仍 fp16 的计算部件）",
     "耗时随 patch 数线性：448×252 的 6 帧 = 4.5 + 23.2 ms×帧（R²=0.997），占 TTFT 19–25%。\n"
     "visual_build 无精度开关 ⇒ 想量化必须回云端重导 ONNX（已实测确认，本地无路）。\n"
     "帧数上限取决于帧形状：112 token/帧 ⇒ 8 帧；154 token/帧（448×352）⇒ 6 帧就到顶\n"
     "（实测最长请求 1015 / maxInputLen 1024，只剩 9 个 token；见图 2 下半）。",
     F_E)

# ================= concat =================
down(LX + 13.0, 78.0, 73.6)
down(LX + 39.5, 78.0, 73.6)
box(LX + 1.0, 68.6, 50.5, 5.0,
    "拼接成一条 token 序列：672 图像 token + 87 文字 token = 759 token（本例）",
    IO_F, IO_E, fs=ANNOT + 0.5)
down(LX + 26.0, 68.6, 65.0)

# ================= decoder stack =================
DY, DH = 26.0, 39.0
box(LX + 1.0, DY, 50.5, DH, "", Q_F, Q_E, z=2)
ax.text(LX + 26.0, DY + DH - 2.6,
        "LLM Decoder × 36 层　—　本项目量化的全部对象",
        ha="center", va="center", fontsize=BASE, fontweight="bold", color=Q_E)

# ---- deepstack: 3 side features from the ViT into the first 3 layers -----
ax.annotate("", xy=(LX + 6.0, DY + DH - 5.0), xytext=(LX + 6.0, 78.0),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color=DS_E,
                            linestyle="--", shrinkA=0, shrinkB=0))
ax.text(LX + 6.9, 67.9, "DeepStack：ViT 第 8/16/24 层的 3 路特征\n注入 decoder 前 3 层（见图 1）",
        ha="left", va="center", fontsize=ANNOT - 0.5, color=DS_E)

box(LX + 3.5, DY + 29.4, 45.5, 3.4, "RMSNorm", INNER, "#c7d2df", fs=ANNOT, z=3)
box(LX + 3.5, DY + 18.6, 45.5, 9.8,
    "自注意力 Attention（GQA 32 Q 头 / 8 KV 头 · headDim 128）\n"
    "q 4096×4096 · k, v 4096×1024 · o 4096×4096 　+ RoPE →  KV cache",
    INNER, "#c7d2df", fs=ANNOT, z=3)
box(LX + 3.5, DY + 13.6, 45.5, 3.4, "RMSNorm", INNER, "#c7d2df", fs=ANNOT, z=3)
box(LX + 3.5, DY + 4.0, 45.5, 8.6,
    "前馈网络 MLP（SwiGLU）\ngate, up 4096×12288 · down 12288×4096",
    INNER, "#c7d2df", fs=ANNOT, z=3)
ax.text(LX + 26.0, DY + 1.9,
        f"每层 7 个线性层 → 36 层共 {quant_w/1e9:.2f} B 参数 = 全部矩阵乘参数的 "
        f"{P_ATTN + P_MLP:.1f}%（注意力 {P_ATTN:.1f}% + MLP {P_MLP:.1f}%）",
        ha="center", va="center", fontsize=ANNOT, color=Q_E, zorder=4)

tag(LX + 48.4, DY + DH - 2.6, "B", Q_E)
note(62.5, "B", "§1.1–1.4　量化：INT4 AWQ (W4A16) / INT8 SmoothQuant (W8A8)",
     "这 36×7 个线性层是全部提速与缩体积的来源：引擎 15.15→4.85 GB、decode 快 2.66×。\n"
     "decode 里它们跑成 W4A16 GEMV kernel = 72.1% 的时间、71.6% 的字节（§1.5）。\n"
     "Roofline：decode 算术强度仅 1.0–3.2 ⇒ 纯访存墙，加速上限就是字节比 3.16×（§1.4）。",
     Q_E)

tag(LX + 47.0, DY + 31.1, "C", "#64748b")
note(53.8, "C", "§1.5　RMSNorm / RoPE / SwiGLU：已被 TensorRT 融合",
     "六个 __myl_* 融合 kernel 合计仅 3.2% 时间，且 CUDA graph 已启用\n"
     "⇒「算子融合 + 降 launch 开销」这条常规优化路线在此没有空间（诚实的负结果）。",
     "#64748b")

tag(LX + 47.0, DY + 26.4, "D", "#64748b")
note(46.6, "D", "§1.1　注意力与 KV cache：非瓶颈（实测）",
     "kernel_mha 只占 decode 时间 2.7%；上下文 128→4000（涨 31×）吞吐仅掉 9%。",
     "#64748b")

tag(LX + 47.0, DY + 11.0, "E", Q_E)
note(41.6, "E", "§1.6　手写 W4A16 GEMV 对照 —— 已证无余量",
     "先实测这台 Orin 的可达读带宽 = 152.3 GB/s（理论 204.8 的 74.4%）；\n"
     "TRT 的 GEMV 已达 147.9 GB/s = 可达值的 97.1%，我最好的手写版只到 52.0%。\n"
     "⇒ 杠杆不在 kernel，而在字节数（位宽 / KV cache / 稀疏）。", Q_E)

# ================= final norm + lm_head =================
down(LX + 26.0, DY, 22.4)
box(LX + 1.0, 18.4, 50.5, 3.8, "最终 RMSNorm", N_F, N_E, fs=ANNOT)
down(LX + 26.0, 18.4, 13.4)
box(LX + 1.0, 8.0, 50.5, 5.4,
    f"输出层 lm_head　4096 × 151936　【fp16，未被量化】\n"
    f"{lm_head/1e9:.2f} B 参数 = 矩阵乘参数的 {P_LMH:.1f}%，却占 decode 字节 {B_LMH:.1f}%",
    F_F, F_E, fs=ANNOT + 0.5, bold=False)
tag(LX + 48.4, 11.7, "F", F_E)
note(13.2, "F", "§1.5 发现 → 优化 A（已实测 +19.6%，2026-07-28）",
     "AWQ 默认跳过输出层 ⇒ 它独占 1.245 GB / token 的搬运量、22.0% 的 decode 时间。\n"
     "量化掉它把每 token 字节 4.853→3.932 GB（实测引擎 3.923 GB，误差 −0.23%），\n"
     "TPOT 30.73→25.69 ms（+19.6%，事前预测 +19%）；代价 n=210 正确率 −1.4 pts。\n"
     "前 8 次云端失败，第 9 次靠把校准 batch 16→1 才成（见 PROBLEMS G3）。", F_E)

# ================= sampling / output =================
down(LX + 26.0, 8.0, 4.4)
box(LX + 1.0, 0.4, 50.5, 4.0,
    "采样 → 1 个 token → 回灌下一步（自回归），直到 EOS 或长度上限 ⇒ 输出一段文字",
    IO_F, IO_E, fs=ANNOT + 0.5)
ax.annotate("", xy=(LX + 0.6, DY + DH / 2), xytext=(LX + 0.6, 2.4),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#94a3b8",
                            connectionstyle="arc3,rad=0.0"))
ax.text(LX - 0.4, DY - 6.0, "自回归回灌", ha="center", va="center",
        fontsize=ANNOT, color="#94a3b8", rotation=90)

# ---- per-token byte budget: ties the module shares back to the wall ------
bx, by, bw, bh = RX + 1.0, 18.0, 42.0, 15.6
ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
             boxstyle="round,pad=0,rounding_size=1.0", mutation_aspect=0.55,
             linewidth=0.8, edgecolor="#94a3b8", facecolor="#fbfcfe"))
ax.text(bx + 1.6, by + bh - 1.9,
        "每生成 1 个 token，必须把整个引擎搬一遍（decode = 访存墙）",
        ha="left", va="center", fontsize=ANNOT, color=INK, fontweight="bold")
budget = [
    ("36×7 线性层 · INT4 权重", "3.473 GB", "71.6%", "72.1%", Q_E),
    ("AWQ scales + zeros", "0.136 GB", "2.8%", "—", Q_E),
    ("lm_head · fp16（可选 INT4）", "1.245 GB", "25.6%", "22.0%", F_E),
]
ax.text(bx + 1.6, by + bh - 4.6, "部件", fontsize=ANNOT, color="#64748b")
for lbl, xx in (("字节/token", 21.5), ("占字节", 29.6), ("占时间", 36.4)):
    ax.text(bx + xx, by + bh - 4.6, lbl, fontsize=ANNOT, color="#64748b",
            ha="right")
for i, (nm, gb, pb, pt, col) in enumerate(budget):
    y = by + bh - 6.6 - i * 2.2
    ax.text(bx + 1.6, y, nm, fontsize=ANNOT, color=col, va="center")
    for val, xx in ((gb, 21.5), (pb, 29.6), (pt, 36.4)):
        ax.text(bx + xx, y, val, fontsize=ANNOT, color="#333333",
                va="center", ha="right")
ax.plot([bx + 1.6, bx + bw - 1.6], [by + 3.0] * 2, "-", color="#cbd5e1", lw=0.7)
ax.text(bx + 1.6, by + 1.6,
        "合计 4.853 GB ÷ 实测 TPOT 33.1 ms = 146.5 GB/s = 可达带宽 152.3 的 96%",
        fontsize=ANNOT, color=INK, va="center")

# ================= legend =================
handles = [
    Line2D([], [], marker="s", ls="", ms=7, mfc=Q_F, mec=Q_E,
           label="本项目量化的模块（INT4 / INT8）"),
    Line2D([], [], marker="s", ls="", ms=7, mfc=F_F, mec=F_E,
           label="仍是 fp16 —— 已定位的优化点"),
    Line2D([], [], marker="s", ls="", ms=7, mfc=N_F, mec=N_E,
           label="无权重 / 不在 LLM 引擎内"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Cosmos-Reason2-8B structure (measured config) and the module each "
             "experiment targets", y=0.995, fontsize=TITLE)
fig.tight_layout(rect=(0, 0.025, 1, 0.975))
fig.savefig(os.path.join(OUT, "model_arch.png"), bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("wrote", os.path.join(OUT, "model_arch.png"))

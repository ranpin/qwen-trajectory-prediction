#!/usr/bin/env python3
"""Upstream Cosmos-Reason2-8B architecture next to the form we actually deploy.

Reason this exists: readers could not map the deployment-side figure
(model_arch.png) onto the model as its authors publish it, and the deployment
figure hid two real parts of the architecture -- the DeepStack multi-scale
visual injection, and the fact that the token-embedding lookup is a runtime
kernel outside the TRT graph rather than a layer of the engine.

Every value is read from artefacts on the device, not from memory:
  engines/int4/visual/config.json   qwen3_vl_vision: depth 27, hidden 1152,
                                    heads 16, FFN 4304, patch 16, merge 2,
                                    temporal patch 2, out_hidden 4096,
                                    deepstack_visual_indexes [8,16,24]
  engines/int4/llm/config.json      qwen3_vl_text: 36 layers, hidden 4096,
                                    GQA 32/8, headDim 128, FFN 12288,
                                    mrope_section [24,20,20], theta 5e6,
                                    num_deepstack_features 3, vocab 151936,
                                    builder max_input_len 1024, max_batch 4
  file sizes                        visual.engine 1.168 GB, llm.engine 4.847 GB,
                                    embedding.safetensors 1.245 GB
  run log                           tokenizer: 151643 vocab + 26 special tokens
  TensorRT-Edge-LLM source          cpp/runtime/preprocess/embeddingPreprocessor
                                    .cpp -> kernel::embeddingLookupWithImage-
                                    Insertion writes PipelineIO::inputsEmbeds,
                                    which is what the LLM engine consumes

Parameter account (self-check): 0.58 B vision + 6.95 B decoder + 0.62 B token
embedding + 0.62 B lm_head = 8.77 B, the checkpoint size NVIDIA publishes.

Run:  .venv/bin/python alpamayo-edge/eval/arch_upstream.py
Out:  alpamayo-edge/docs/figures/arch_upstream.png
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

VIS_F, VIS_E = "#fef3c7", "#d97706"    # kept fp16
LLM_F, LLM_E = "#dbeafe", "#2563eb"    # quantized by us
HOST_F, HOST_E = "#f1f5f9", "#64748b"  # host-side / runtime, not a TRT engine
DS_F, DS_E = "#f3e8ff", "#7c3aed"      # deepstack path
IO_F, IO_E = "#eef2f7", "#94a3b8"

fig, ax = plt.subplots(figsize=(10.4, 7.8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.grid(False)

LX, LW = 3.0, 49.0     # left panel (upstream)
RX, RW = 55.5, 43.5    # right panel (deployed)
SMALL = ANNOT - 0.5


def box(x, y, w, h, text, face, edge, fs=ANNOT, bold=False, z=3, lsp=1.5,
        ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=0.9", mutation_aspect=0.55,
                 linewidth=0.9, edgecolor=edge, facecolor=face, zorder=z))
    ax.text(x + (w / 2 if ha == "center" else 1.4), y + h / 2, text,
            ha=ha, va="center", fontsize=fs, color=INK, zorder=z + 1,
            fontweight="bold" if bold else "normal", linespacing=lsp)


def arrow(pts, color="#475569", lw=1.1, ls="-"):
    """Poly-line arrow through a list of (x, y) points."""
    for a, b in zip(pts, pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], ls=ls, color=color, lw=lw, zorder=2)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                linestyle=ls, shrinkA=0, shrinkB=0))


def tag(x, y, mark, color):
    ax.text(x, y, mark, ha="center", va="center", fontsize=SMALL, color=color,
            fontweight="bold", zorder=8,
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="white",
                      edgecolor=color, linewidth=0.9))


ax.text(LX + LW / 2, 97.6, "(a) 上游发布的模型结构", ha="center", va="center",
        fontsize=BASE, fontweight="bold", color="#334155")
ax.text(LX + LW / 2, 94.6,
        "Cosmos-Reason2-8B = NVIDIA 在 Qwen3-VL-8B-Instruct 上后训练；数值全取自实测 config",
        ha="center", va="center", fontsize=SMALL, color="#64748b")
ax.text(RX + RW / 2, 97.6, "(b) 我们部署后的实际形态", ha="center", va="center",
        fontsize=BASE, fontweight="bold", color="#334155")
ax.text(RX + RW / 2, 94.6,
        "TensorRT-Edge-LLM v0.9.0 · Jetson Orin sm_87；圈号与左图模块一一对应",
        ha="center", va="center", fontsize=SMALL, color="#64748b")

# ============================ (a) upstream ============================
box(LX, 86.8, 23.0, 5.2, "图像 / 视频帧（H×W×3）", IO_F, IO_E)
box(LX + 26.0, 86.8, 23.0, 5.2, "文字 prompt（自然语言）", IO_F, IO_E)
arrow([(LX + 11.5, 86.8), (LX + 11.5, 84.2)])
arrow([(LX + 37.5, 86.8), (LX + 37.5, 84.2)])

box(LX, 71.0, 23.0, 13.0,
    "视觉编码器 ViT（qwen3_vl_vision）\n\n"
    "27 层 · d=1152 · 16 头 · FFN 4304\n"
    "patch 16×16 · 时间 patch 2\nGELU-tanh · 位置嵌入 2304\n≈0.58 B 参数",
    VIS_F, VIS_E)
tag(LX + 21.0, 82.4, "1", VIS_E)

box(LX + 26.0, 78.6, 23.0, 5.4,
    "Qwen3 BPE 分词器（tokenizer.json）\n151 643 词 + 26 个特殊 token",
    HOST_F, HOST_E)
box(LX + 26.0, 71.0, 23.0, 5.6,
    "词嵌入表  151936 × 4096（fp16）\n查表 → 每个文字 token 一个 4096 维向量",
    HOST_F, HOST_E)
tag(LX + 47.0, 75.2, "3", HOST_E)
arrow([(LX + 37.5, 78.6), (LX + 37.5, 76.6)])

box(LX, 63.0, 23.0, 6.2,
    "2×2 spatial merge + Merger MLP\n→ 每 token 4096 维（对齐文本宽度）",
    VIS_F, VIS_E)
arrow([(LX + 11.5, 71.0), (LX + 11.5, 69.2)])

box(LX + 26.0, 61.2, 23.0, 8.0,
    "DeepStack（Qwen3-VL 特有）\n取 ViT 第 8 / 16 / 24 层的中间特征\n"
    "→ 3 路多尺度视觉特征\n注入解码器前 3 层 (num_deepstack_features=3)",
    DS_F, DS_E)
tag(LX + 47.0, 67.4, "2", DS_E)
arrow([(LX + 23.0, 77.5), (LX + 24.2, 77.5), (LX + 24.2, 65.2), (LX + 26.0, 65.2)],
      color=DS_E, ls="--")

box(LX, 51.0, 34.0, 6.0,
    "按 <|image_pad|> 占位位置，把视觉 token 写进文字 embedding 序列\n"
    "→ 一条混合向量序列（图像与文字 token 等宽、同序）",
    IO_F, IO_E)
tag(LX + 35.8, 54.0, "4", IO_E)
arrow([(LX + 11.5, 63.0), (LX + 14.0, 57.0)])
# text embeddings reach the mix box down the corridor between the two columns,
# so the line does not cross the DeepStack box that sits in the right column
arrow([(LX + 26.0, 73.8), (LX + 25.4, 73.8), (LX + 25.4, 57.0)])

box(LX, 28.0, 49.0, 19.0,
    "文本解码器（qwen3_vl_text）\n\n"
    "36 层 · d=4096 · GQA 32 Q 头 / 8 KV 头 · headDim 128\n"
    "每层：RMSNorm → 注意力 (q,k,v,o) + RoPE + KV cache\n"
    "　　　→ RMSNorm → MLP（SwiGLU，FFN 12288）\n"
    "mRoPE 三段 (t, h, w) = [24, 20, 20] interleaved · θ = 5×10⁶\n"
    "位置上限 262 144 · 6.95 B 参数",
    LLM_F, LLM_E)
tag(LX + 47.0, 45.0, "5", LLM_E)
arrow([(LX + 17.0, 51.0), (LX + 17.0, 47.0)])
arrow([(LX + 41.0, 61.2), (LX + 41.0, 47.0)], color=DS_E, ls="--")

box(LX, 20.0, 49.0, 5.6, "输出层 lm_head　4096 × 151936　→ 下一个 token 的概率分布",
    LLM_F, LLM_E)
tag(LX + 47.0, 22.8, "6", LLM_E)
arrow([(LX + 24.5, 28.0), (LX + 24.5, 25.6)])

box(LX, 12.2, 49.0, 5.6, "采样 → 1 个 token → 回灌（自回归），直到 EOS 或长度上限",
    IO_F, IO_E)
arrow([(LX + 24.5, 20.0), (LX + 24.5, 17.8)])

box(LX, 4.4, 49.0, 5.8,
    "参数账（自校验）：视觉 0.58 B + 解码器 6.95 B + 词嵌入 0.62 B + lm_head 0.62 B\n"
    "= 8.77 B —— 与 NVIDIA 公布的 checkpoint 规模一致",
    "#f8fafc", "#94a3b8")

# ============================ (b) deployed ============================
box(RX, 79.0, RW, 13.0,
    "主机侧（CPU，不含任何 TRT 引擎）\n\n"
    "我们这一步：从 benchmark 视频均匀采 6 帧、长边缩到 448 px\n"
    "Qwen2VLImageProcessor：bicubic 缩到 32 的整数倍 → /255 → 归一化 0.5\n"
    "Qwen3 分词器 + chat template：文字与 <|image_pad|> 占位一起编成 id\n"
    "　（性能基准 §1.1 / §1.4–1.6 不带图，走的就是这条纯文字路径）",
    HOST_F, HOST_E, ha="left")
ax.text(RX + 1.4, 77.2, "像素→token 的完整链条见图 2", ha="left", va="center",
        fontsize=SMALL, color="#94a3b8")

box(RX, 68.4, 34.0, 7.4,
    "visual.engine　fp16　1.168 GB（≈0.584 B 参数）\n"
    "输出：视觉 token（4096 维）＋ 3 路 deepstack 特征",
    VIS_F, VIS_E)
tag(RX + 31.8, 73.8, "1", VIS_E)
tag(RX + 31.8, 70.2, "2", DS_E)
arrow([(RX + 9.0, 79.0), (RX + 9.0, 75.8)])

box(RX, 51.0, 34.0, 12.2,
    "运行时 GPU kernel（不在 TRT 图内）\n\n"
    "embeddingLookupWithImageInsertion：\n"
    "　· 文字 token id → 查 embedding.safetensors\n"
    "　· 图像占位位置 → 写入视觉 token\n"
    "输出 inputsEmbeds —— 这才是 LLM 引擎的输入",
    HOST_F, HOST_E, ha="left")
tag(RX + 31.8, 60.8, "3", HOST_E)
tag(RX + 31.8, 56.6, "4", IO_E)
arrow([(RX + 9.0, 68.4), (RX + 9.0, 63.2)])

box(RX, 31.4, 34.0, 16.0,
    "llm.engine　INT4 4.847 GB / INT8 8.30 GB\n\n"
    "输入 = inputsEmbeds（不是 token id）\n"
    "　　＋ 3 路 deepstack 特征 ＋ 位置 / KV 张量\n"
    "内含 = 36 层 decoder ＋ 最终 RMSNorm ＋ lm_head\n"
    "　　（36×7 个线性层已量化；lm_head 仍是 fp16，\n"
    "　　 AWQ 默认跳过输出层 —— 见图 3 标记 F）",
    LLM_F, LLM_E, ha="left")
tag(RX + 31.8, 44.4, "5", LLM_E)
tag(RX + 31.8, 40.0, "6", LLM_E)
arrow([(RX + 9.0, 51.0), (RX + 9.0, 47.4)])
arrow([(RX + 34.0, 71.5), (RX + 37.6, 71.5), (RX + 37.6, 44.0), (RX + 34.0, 44.0)],
      color=DS_E, ls="--")

box(RX, 23.6, 34.0, 6.0,
    "采样 → token → 分词器解码 → 文字\n（回灌下一步，直到 EOS 或长度上限）", IO_F, IO_E)
arrow([(RX + 9.0, 31.4), (RX + 9.0, 29.6)])

box(RX, 3.2, RW, 18.6,
    "与上游结构的三处差异（都是部署造成的）\n\n"
    "① 36×7 个线性层被量化：INT4(W4A16) / INT8(W8A8)；\n"
    "　 视觉编码器仍是 fp16；lm_head 默认 fp16，可选 INT4（已实测 +19.6%）\n"
    "② 词嵌入查表被搬出 TRT 图 → 独立权重文件 + 一个 kernel，\n"
    "　 这样才能把视觉向量插进同一个 buffer（上游在 PyTorch\n"
    "　 里就是 index_select + masked_scatter 两句）\n"
    "③ 引擎构建时固定 maxInputLen=1024 / maxBatch=4 —— 模型\n"
    "　 本身支持 262 144 上下文，是引擎参数把它限住了",
    "#fff7ed", "#c2410c", ha="left")

handles = [
    Line2D([], [], marker="s", ls="", ms=7, mfc=LLM_F, mec=LLM_E,
           label="我们量化的部分（INT4 / INT8）"),
    Line2D([], [], marker="s", ls="", ms=7, mfc=VIS_F, mec=VIS_E,
           label="保持 fp16 的视觉部分"),
    Line2D([], [], marker="s", ls="", ms=7, mfc=HOST_F, mec=HOST_E,
           label="主机侧 / 运行时（不在 TRT 引擎内）"),
    Line2D([], [], marker="s", ls="", ms=7, mfc=DS_F, mec=DS_E,
           label="DeepStack 多尺度视觉注入"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.008))
fig.suptitle("Published architecture (left) vs the deployed decomposition "
             "(right); circled numbers pair the two", y=0.996, fontsize=TITLE)
fig.tight_layout(rect=(0, 0.028, 1, 0.972))
fig.savefig(os.path.join(OUT, "arch_upstream.png"), bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("wrote", os.path.join(OUT, "arch_upstream.png"))

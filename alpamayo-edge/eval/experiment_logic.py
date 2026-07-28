#!/usr/bin/env python3
"""Why each experiment exists: the question -> experiment -> conclusion chain.

A reader's second complaint after "what is the I/O" is "what is the logic that
connects these experiments". The eight result subsections were not a checklist;
each one exists because the previous one's conclusion raised a specific question.
This figure draws that chain, with the transition between rounds labelled by the
question it raised.

Nothing here is new data — every conclusion box quotes a number already measured
and reported in the corresponding subsection (§1.1-§1.8). Colour marks the kind
of outcome: confirmed / negative result (headroom proven absent) / not achieved.

Run:  .venv/bin/python alpamayo-edge/eval/experiment_logic.py
Out:  alpamayo-edge/docs/figures/experiment_logic.png
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

# outcome palette
OK_F, OK_E = "#dcfce7", "#16a34a"      # answered / confirmed
NEG_F, NEG_E = "#e8eaf6", "#5c6bc0"    # negative result: headroom proven absent
NO_F, NO_E = "#fef3c7", "#d97706"      # attempted, not achieved -> prediction
Q_F, Q_E = "#f8fafc", "#64748b"        # the question
E_F, E_E = "#dbeafe", "#2563eb"        # the experiment

fig, ax = plt.subplots(figsize=(10.4, 9.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.grid(False)

QX, QW = 2.6, 22.0
EX, EW = 26.0, 26.5
CX, CW = 54.0, 45.5
H = 7.8


def panel(x, y, w, h, text, face, edge, fs=ANNOT, head=None, hcol=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=0.9", mutation_aspect=0.55,
                 linewidth=0.9, edgecolor=edge, facecolor=face, zorder=2))
    ty = y + h - 1.5
    if head:
        ax.text(x + 1.3, ty, head, ha="left", va="top", fontsize=fs,
                color=hcol or edge, fontweight="bold", zorder=3)
        ty -= 2.25
    ax.text(x + 1.3, ty, text, ha="left", va="top", fontsize=fs, color=INK,
            zorder=3, linespacing=1.55)


def band(ybot, num, q, sec, exp, chead, ctext, cface, cedge):
    """One round: question -> experiment -> conclusion."""
    panel(QX, ybot, QW, H, q, Q_F, Q_E, head=f"{num}　问题", hcol=Q_E)
    panel(EX, ybot, EW, H, exp, E_F, E_E, head=sec, hcol=E_E)
    panel(CX, ybot, CW, H, ctext, cface, cedge, head=chead, hcol=cedge)
    for x0, x1 in ((QX + QW, EX), (EX + EW, CX)):
        ax.annotate("", xy=(x1 - 0.15, ybot + H / 2), xytext=(x0 + 0.15, ybot + H / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#94a3b8",
                                    shrinkA=0, shrinkB=0))


def link(y, text, color="#475569"):
    """The logical transition: what the previous conclusion made us ask next."""
    ax.annotate("", xy=(QX + 5.0, y - 1.35), xytext=(QX + 5.0, y + 1.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color,
                                shrinkA=0, shrinkB=0))
    ax.text(QX + 6.6, y, text, ha="left", va="center", fontsize=ANNOT,
            color=color, style="italic")


# ---- column headers ------------------------------------------------------
for x, w, t in ((QX, QW, "提出的问题"), (EX, EW, "做的实验（对应章节）"),
                (CX, CW, "得到的结论（全部为实测）")):
    ax.text(x + w / 2, 97.6, t, ha="center", va="center", fontsize=BASE,
            fontweight="bold", color="#334155")

# ---- rounds --------------------------------------------------------------
band(88.0, "①",
     "8B 的多模态模型，能不能在\n一块 Jetson Orin 上跑起来？",
     "§1.1　量化 → 建引擎 → 服务指标",
     "AWQ / SmoothQuant 量化，Orin 本地\n建 TRT 引擎，测 TTFT/TPOT/TPS+功耗",
     "能，而且量化是唯一可部署方案",
     "INT4 引擎 4.85 GB、30.9 tok/s、E2E 快 2.5×；\n"
     "FP16(15 GB) 在 32 GB 机上 build(峰 55 GB) 与载入双双 OOM。",
     OK_F, OK_E)

link(85.6, "跑起来了 —— 但权重被压到 4 bit，它还答得对吗？")

band(77.2, "②",
     "量化之后任务正确率掉多少？\n掉的是不是统计显著的？",
     "§1.2 官方带标注基准 · §1.3 探针",
     "Cosmos-Reason1-Benchmark n=210 选\n择题，三精度同帧；另 12 条 prompt",
     "在 6.7 pts 分辨率内未发现损失（MDE 见 §1.2）",
     "76.2 / 75.2 / 73.8%（FP16 / INT8 / INT4），\n"
     "McNemar p > 0.4、Wilson 95% CI 全部重叠。",
     OK_F, OK_E)

link(74.8, "既然精度没问题，回头看性能：那三处反常的加速比是怎么来的？")

band(66.4, "③",
     "INT4 的 prefill 为何不快于\nFP16？decode 为何只 2.66×？",
     "§1.4　Roofline 分析",
     "把实测工作点画到 Orin 的屋顶线上\n（带宽屋顶 / 算力屋顶，GPU 稠密值）",
     "两个阶段撞的是两面不同的墙",
     "decode 算术强度仅 1.0–3.2 ⇒ 访存墙，上限 = 字节比 3.16×；\n"
     "prefill 是计算墙，且 W4A16 要反量化 ⇒ 上限就是 FP16。",
     OK_F, OK_E)

link(64.0, "既然 decode 撞的是访存墙，那这些字节具体花在哪个 kernel、哪个模块？")

band(55.6, "④",
     "decode 的时间与字节，逐\nkernel 看分别花在哪里？",
     "§1.5　Nsight Systems 逐 kernel 剖析",
     "真机 nsys（必须加 --cuda-graph-trace\n=node），并与解析字节互校 +0.13%",
     "找到一处主要开销 + 一条常规路线被否定",
     "W4A16 GEMV 占 72.1% 时间；lm_head 竟未被量化，独占\n"
     "25.6% 字节 / 22.0% 时间；融合算子合计仅 3.2%（无空间）。",
     OK_F, OK_E)

link(53.2, "于是出现两条互斥的提速路线 —— 提高带宽效率 还是 继续压字节？两条都验证：",
     "#b45309")

band(44.8, "⑤A",
     "路线 B：自己写 kernel，把\nGEMV 的带宽效率提上去？",
     "§1.6　先测上限，再写 GEMV",
     "自写流式读 kernel 实测这台机器的\n可达读带宽，再与 TRT 逐形状对照",
     "此路不通 —— 我自己的预测被自己的测量推翻",
     "可达读带宽只有 152.3 GB/s（理论 204.8 的 74.4%），\n"
     "TRT 已达其 97.1% ⇒ 余量 3%，主动放弃（在原处标注推翻）。",
     NEG_F, NEG_E)

# fork: rounds 5A and 5B are two parallel routes out of round 4's conclusion
FKX = 1.1
ax.plot([QX + 5.0, FKX], [52.3, 52.3], "-", color="#b45309", lw=1.0, zorder=1)
ax.plot([FKX, FKX], [52.3, 37.9], "-", color="#b45309", lw=1.0, zorder=1)
ax.annotate("", xy=(QX - 0.15, 37.9), xytext=(FKX, 37.9),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#b45309",
                            shrinkA=0, shrinkB=0))

band(34.0, "⑤B",
     "路线 A：把 lm_head 也量化\n掉，省掉 19% 的字节？",
     "§1.5 框　云端 9 次尝试，第 9 次成",
     "前 8 次调显存/模块放置全失败；第 9 次\n"
     "把报错数字因式分解，只改校准 batch 16→1",
     "成了：decode +19.6%（事前预测 +19%），代价正确率 −1.4 pts",
     "瓶颈是校准批次的 logits 张量 = batch×seq×vocab×4B，与权重\n"
     "大小无关。8B 有效带宽升至 153.1 ≈ 可达上限 152.3 GB/s ⇒ 饱和。",
     OK_F, OK_E)

link(31.6, "以上性能全部只测了 LLM —— 可它是个 VLM，真实请求还带着图像。")

band(23.2, "⑥",
     "真实的多模态负载有多贵？\n多路相机能撑到几帧？",
     "§1.7　视觉编码器按帧数扫描（1/2/4/6）",
     "llm_inference --dumpProfile 取 TRT 分\n段 GPU 时间，拟合并外推 8 路相机",
     "被忽略的四分之一，外加一道硬墙",
     "视觉编码器 = 4.5 + 23.2 ms×帧（R²=0.997），占 TTFT 19–25%；\n"
     "帧数上限随帧形状变（112 vs 154 tok/帧）；已重建 maxInputLen=4096。",
     OK_F, OK_E)

link(20.8, "精度这条轴已经调到头了 —— 那换一个更小的模型，是不是更划算？")

band(12.4, "⑦",
     "部署到底该缩精度，还是缩\n模型规模？",
     "§1.8　2B vs 8B 的 Pareto",
     "2B 用与已部署 8B 完全相同的配置量\n化，唯一变量是规模，同一打分器",
     "本项目最可执行的一条选型结论",
     "8B 三个精度点几乎水平（p > 0.4）；INT4 2B 掉到 65.2%\n"
     "（−11.0 pts，p = 0.005 显著）⇒ 激进量化 + 保留大模型。",
     OK_F, OK_E)

# ---- closing strip -------------------------------------------------------
ax.annotate("", xy=(QX + 5.0, 8.4), xytext=(QX + 5.0, 11.0),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#475569",
                            shrinkA=0, shrinkB=0))
panel(QX, 0.6, 96.9, 7.6,
      "① 交互式 / 长输出选 INT4（vs FP16：decode 2.66×、能效 3.1×）；批量 / 长 prompt 选 INT8"
      "（vs FP16：prefill 1.87×、能效 2.1×）；② 32 GB 机上 FP16 不可行 ⇒ 量化是可部署性的前提；\n"
      "③ lm_head 量化已兑现（+19.6%，代价 −1.4 pts）⇒ **decode 这一条**已达可达带宽上限；已证死的只有 W4A8（工具里不存在）与手写 kernel（实测无余量）。\n"
      "**ViT / vocab reduction / 投机解码三条仍未探明**——此前把「工具开关只给 fp8」误当成「不可行」。",
      "#f8fafc", "#94a3b8",
      head="⇒ 以上各轮实验合起来给出的部署决策（§三 思考讨论）", hcol=INK)

handles = [
    Line2D([], [], marker="s", ls="", ms=7, mfc=OK_F, mec=OK_E,
           label="结论已被实测确认"),
    Line2D([], [], marker="s", ls="", ms=7, mfc=NEG_F, mec=NEG_E,
           label="负结果：余量被证明不存在（含推翻自己的预测）"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.004))
fig.suptitle("Experiment chain: each round's conclusion is the next round's "
             "question", y=0.996, fontsize=TITLE)
fig.tight_layout(rect=(0, 0.022, 1, 0.978))
fig.savefig(os.path.join(OUT, "experiment_logic.png"), bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("wrote", os.path.join(OUT, "experiment_logic.png"))

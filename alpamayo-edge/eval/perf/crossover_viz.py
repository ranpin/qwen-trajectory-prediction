#!/usr/bin/env python3
"""INT4 vs INT8: which precision wins depends on the input/output token ratio.

Out: docs/figures/precision_crossover.png
Run: .venv/bin/python alpamayo-edge/eval/perf/crossover_viz.py

Data: eval/perf/len4096_sweep.csv (orin-dog, 2026-07-27, MAXN, batch 1).
Both curves come from the *matched* len4096 engine pair (same build config, same TRT
opt shape 2048) so the comparison is apples-to-apples across 128..4096 input tokens.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paper_style import apply, C4, C8, INK, ANNOT, LEGEND, TICK, TITLE, LABEL, WIDTH_FULL

apply()
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(os.path.join(HERE, "len4096_sweep.csv"))))


def get(engine, mode, length):
    for r in rows:
        if r["engine"] == engine and r["mode"] == mode and int(r["len"]) == length:
            return float(r["e2e_ms"])
    raise KeyError((engine, mode, length))


LENS = [128, 512, 1024, 2048, 4096]
pre4 = {n: get("int4_len4096", "prefill", n) for n in LENS}
pre8 = {n: get("int8_len4096", "prefill", n) for n in LENS}
tpot4 = get("int4_len4096", "decode", 512)
tpot8 = get("int8_len4096", "decode", 512)
PEN = tpot8 - tpot4                      # INT8 pays this per output token
BREAK = {n: (pre4[n] - pre8[n]) / PEN for n in LENS}   # break-even output length
# The ratio is stable (82-84) for n_in >= 512; at 128 fixed per-call overhead dominates
# and INT8's prefill saving nearly vanishes, so that point is excluded from the fit.
FIT = [n for n in LENS if n >= 512]
RATIO = sum(n / BREAK[n] for n in FIT) / len(FIT)      # n_in / n_out at break-even

print(f"TPOT  INT4 {tpot4:.2f} ms | INT8 {tpot8:.2f} ms | INT8 penalty {PEN:.2f} ms/token")
for n in LENS:
    print(f"  in={n:5}  prefill INT4 {pre4[n]:8.1f}  INT8 {pre8[n]:8.1f}  "
          f"saving {pre4[n]-pre8[n]:7.1f} ms  break-even out={BREAK[n]:6.2f} tok  "
          f"n_in/n_out={n/BREAK[n]:.1f}")
print(f"=> boundary: INT8 wins when n_out < n_in / {RATIO:.0f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 3.9))

# ---- (a) E2E vs output length, at two input lengths ----------------------
outs = list(range(0, 161))
for n_in, ls, al, lw in ((512, "-", 0.42, 1.2), (4096, "-", 1.0, 1.8)):
    a1.plot(outs, [(pre4[n_in] + m * tpot4) / 1000 for m in outs], ls, color=C4, lw=lw,
            alpha=al, label=f"INT4, {n_in} in")
    a1.plot(outs, [(pre8[n_in] + m * tpot8) / 1000 for m in outs], ls, color=C8, lw=lw,
            alpha=al, label=f"INT8, {n_in} in")
    xb = BREAK[n_in]
    a1.plot([xb], [(pre4[n_in] + xb * tpot4) / 1000], "o", ms=5, mfc="white",
            mec=INK, mew=1.2, zorder=5)
    a1.annotate(f"交叉 {xb:.0f} tok", xy=(xb, (pre4[n_in] + xb * tpot4) / 1000),
                xytext=(xb + 8, (pre4[n_in] + xb * tpot4) / 1000 - 0.55),
                fontsize=ANNOT, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=INK))
a1.set_xlabel("输出 token 数", fontsize=LABEL)
a1.set_ylabel("端到端延迟 (s)", fontsize=LABEL)
a1.set_title("端到端延迟 vs 输出长度 (orin-dog, MAXN, batch 1)", fontsize=TITLE)
a1.legend(fontsize=LEGEND, frameon=False, ncol=2, loc="upper left")
a1.tick_params(labelsize=TICK)
a1.set_xlim(0, 160)

# ---- (b) region map ------------------------------------------------------
a2.set_xscale("log"); a2.set_yscale("log")
a2.set_xlim(100, 6000); a2.set_ylim(0.8, 400)
xs = [100, 6000]
a2.plot(xs, [x / RATIO for x in xs], "-", color=INK, lw=1.4, zorder=4)
a2.fill_between([100, 6000], [0.8, 0.8], [100 / RATIO, 6000 / RATIO],
                color=C8, alpha=0.13, zorder=1)
a2.fill_between([100, 6000], [100 / RATIO, 6000 / RATIO], [400, 400],
                color=C4, alpha=0.13, zorder=1)
a2.text(3400, 1.35, "INT8 更快", fontsize=ANNOT + 0.5, color="#a16207", ha="center")
a2.text(300, 150, "INT4 更快", fontsize=ANNOT + 0.5, color="#1d4ed8", ha="center")
a2.text(5200, 5200 / RATIO * 1.35, f"$n_{{out}}=n_{{in}}/{RATIO:.0f}$",
        fontsize=ANNOT, color=INK, ha="right", rotation=17)

PTS = [
    (759, 1, "n=210 基准\n实测 INT8 快 1.43x", 10, 12),
    (512, 128, "已发布 E2E (512 in / 128 out)\n实测 INT4 快 1.49x", -8, -42),
    (2130, 12, "1 帧原生 1080p\n+ 选择题作答", 10, 16),
    (759, 110, "6 帧 + 场景描述", 14, 30),
]
for x, y, lab, dx, dy in PTS:
    a2.plot([x], [y], "*", ms=9, color=INK, zorder=6)
    a2.annotate(lab, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=ANNOT, color=INK, ha="left" if dx > 0 else "right",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=INK))
a2.set_xlabel("输入 token 数（图像 + 文字）", fontsize=LABEL)
a2.set_ylabel("输出 token 数", fontsize=LABEL)
a2.set_title("按负载形状划分的精度选择 (orin-dog, batch 1)", fontsize=TITLE)
a2.tick_params(labelsize=TICK)
a2.set_xticks([128, 512, 1024, 2048, 4096])
a2.set_xticklabels(["128", "512", "1k", "2k", "4k"])
a2.set_yticks([1, 10, 100])
a2.set_yticklabels(["1", "10", "100"])
a2.grid(True, which="major", axis="both", ls=":", lw=0.5, alpha=0.5)

fig.tight_layout()
p = os.path.join(OUT, "precision_crossover.png")
fig.savefig(p, bbox_inches="tight", facecolor="white")
print("wrote", p)

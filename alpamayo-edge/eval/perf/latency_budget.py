#!/usr/bin/env python3
"""AV 延迟预算：这套系统在真实相机分辨率下能跑多少 Hz？

把 §1.1/§1.7/§1.9 的实测数字组合成部署判断，而不是再堆一组百分比。
Out: eval/perf/latency_budget.json（表格数据）
Run: .venv/bin/python alpamayo-edge/eval/perf/latency_budget.py

三个实测输入，全部来自本仓库已入库的测量：
  1. prefill TTFT vs 输入 token —— eval/perf/len4096_sweep.csv（len4096 引擎对，10 次迭代）
  2. 视觉编码器 —— §1.7：4.5 + 23.2 ms/帧 @ 448 patch/帧 ⇒ 0.0518 ms/patch（线性 R²=0.997）
  3. decode TPOT —— 200 次迭代：INT4 30.65 / INT8 49.75 ms
口径：orin-dog（AGX Orin 32GB, sm_87, MAXN），batch=1。token 数 = (W/32)×(H/32)，patch 数 = 4×token。
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MS_PER_PATCH = 23.2 / 448          # §1.7 斜率 ÷ 每帧 patch 数
VISION_FIXED = 4.5                 # §1.7 截距
TPOT = {"int4": 30.65, "int8": 49.75}

rows = list(csv.DictReader(open(os.path.join(HERE, "len4096_sweep.csv"))))
PRE = {p: sorted((int(r["len"]), float(r["e2e_ms"]))
                 for r in rows if r["engine"] == f"{p}_len4096" and r["mode"] == "prefill")
       for p in ("int4", "int8")}


def prefill_ms(prec, n):
    """实测点之间线性插值；超出 4096 用 compute-bound 模型外推（并标注）。"""
    pts = PRE[prec]
    if n <= pts[-1][0]:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= n <= x1:
                return y0 + (y1 - y0) * (n - x0) / (x1 - x0), False
        return pts[0][1] * n / pts[0][0], False
    # matmul 2*P*n + attention 144*n^2*d，除以能复现实测的有效算力
    P, L, D = 7.575e9, 36, 4096
    def flops(m): return 2 * P * m + 4 * L * m * m * D
    eff = flops(4096) / (pts[-1][1] / 1000)          # 用 4096 实测点定标
    return flops(n) / eff * 1000, True


def tok(w, h):
    W, H = -(-w // 32) * 32, -(-h // 32) * 32
    return (W // 32) * (H // 32)


SCEN = [
    ("现状基准：1 相机 × 6 帧 @448×252", 6, 448, 252, 1, 87),
    ("6 相机 × 1 帧 @448×256（同分辨率铺开）", 6, 448, 256, 1, 90),
    ("6 相机 × 1 帧 @960×544（视觉引擎每图上限 0.52 MP）", 6, 960, 544, 1, 90),
    ("1 相机 × 1 帧 @1080p（单帧原生）", 1, 1920, 1080, 1, 90),
    ("6 相机 × 1 帧 @1080p（真实车载配置）", 6, 1920, 1080, 1, 90),
    ("现状基准 + 场景描述输出（110 token）", 6, 448, 252, 110, 87),
]

out = {"device": "orin-dog (AGX Orin 32GB, sm_87, MAXN), batch=1",
       "inputs": {"vision_ms_per_patch": MS_PER_PATCH, "vision_fixed_ms": VISION_FIXED,
                  "tpot_ms_200iter": TPOT,
                  "prefill_measured": {p: dict(PRE[p]) for p in PRE}},
       "engine_caps": {"maxInputLen": 4096, "maxImageTokens": 4096,
                       "maxImageTokensPerImage": 2048},
       "scenarios": []}

print(f"{'场景':52}{'img tok':>8}{'总输入':>8}{'视觉':>9}{'INT4 总':>10}{'Hz':>7}"
      f"{'INT8 总':>10}{'Hz':>7}  {'装得下?':>8}")
for name, nf, w, h, n_out, n_txt in SCEN:
    t = tok(w, h)
    img = nf * t
    n_in = img + n_txt
    vis = VISION_FIXED + MS_PER_PATCH * img * 4
    rec = {"scenario": name, "frames": nf, "frame_px": f"{w}x{h}", "tok_per_frame": t,
           "image_tokens": img, "input_tokens": n_in, "output_tokens": n_out,
           "vision_ms": round(vis, 1)}
    fits = n_in <= 4096 and img <= 4096 and t <= 2048
    rec["fits_engine"] = fits
    for p in ("int4", "int8"):
        pf, extrap = prefill_ms(p, n_in)
        tot = vis + pf + n_out * TPOT[p]
        rec[p] = {"prefill_ms": round(pf, 1), "total_ms": round(tot, 1),
                  "hz": round(1000 / tot, 2), "prefill_extrapolated": extrap}
    print(f"{name:52}{img:>8}{n_in:>8}{vis:>8.0f}ms{rec['int4']['total_ms']:>9.0f}ms"
          f"{rec['int4']['hz']:>7.2f}{rec['int8']['total_ms']:>9.0f}ms{rec['int8']['hz']:>7.2f}"
          f"  {'是' if fits else '否(超上限)':>8}")
    out["scenarios"].append(rec)

# 口径自证：把模型算出的基准场景与端到端实测对账
b = out["scenarios"][0]
MEAS = {"int4": 703.0, "int8": 493.0}   # run_int4/int8.log 推理段 ÷ 110 请求（§1.9）
out["validation_vs_measured"] = {
    "scenario": b["scenario"],
    "measured_ms_per_request": MEAS,
    "modelled_ms": {p: b[p]["total_ms"] for p in MEAS},
    "model_underestimates_pct": {p: round(100 * (1 - b[p]["total_ms"] / MEAS[p]), 1) for p in MEAS},
    "reason": ("模型只算 视觉 + prefill + n_out×TPOT，不含分词/词嵌入 gather/请求 IO/主机侧开销。"
               "⇒ 表中的 Hz 是**乐观上界**，真实吞吐比它低 6–12%。")}
print(f"\n口径自证（基准场景）：模型 INT4 {b['int4']['total_ms']:.0f} / INT8 {b['int8']['total_ms']:.0f} ms"
      f" vs 实测 703 / 493 ms ⇒ 低估 "
      f"{out['validation_vs_measured']['model_underestimates_pct']['int4']}% / "
      f"{out['validation_vs_measured']['model_underestimates_pct']['int8']}%（缺主机侧开销）")

# 要达到 10 Hz 需要什么
need = b["int8"]["total_ms"] / 100.0
out["gap_to_10hz"] = {
    "cheapest_scenario_ms": b["int8"]["total_ms"],
    "speedup_needed_for_10hz": round(need, 1),
    "int8_over_int4_actual": round(b["int4"]["total_ms"] / b["int8"]["total_ms"], 2),
    "cosmos_2b_prefill_speedup_measured": 4.02,
    "note": ("即使换 2B（prefill 实测快 4.02x，但正确率掉 11.0 pts 且 McNemar p=0.005 显著），"
             "最快场景也只到约 3.9 Hz。10 Hz 在本硬件 + 本模型量级上不可达。")}
print(f"\n最便宜场景（INT8）{b['int8']['total_ms']:.0f} ms ⇒ 要到 10 Hz 还需 {need:.1f}× 提速；"
      f"量化只给了 {out['gap_to_10hz']['int8_over_int4_actual']}×，2B 也只到约 3.9 Hz。")

json.dump(out, open(os.path.join(HERE, "latency_budget.json"), "w"),
          indent=1, ensure_ascii=False)
print("wrote latency_budget.json")

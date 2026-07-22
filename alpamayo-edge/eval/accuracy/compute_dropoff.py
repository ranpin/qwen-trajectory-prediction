#!/usr/bin/env python3
"""Aggregate the Kaggle FP16 reference (acc_ref.json) into quantization drop-off
metrics: mean perplexity of int4/int8 deployed outputs vs the fp16 reference's
own output (scored by the SAME fp16 model), and token-agreement with the fp16
greedy reference. Prints a markdown table ready to paste into the docs."""
import json, sys

P = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kgl_out/acc_ref.json"
d = json.load(open(P))
R = d["results"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


ppl_fp16 = mean([r["ppl_fp16_own"] for r in R])
ppl_i4 = mean([r["ppl_int4"] for r in R])
ppl_i8 = mean([r["ppl_int8"] for r in R])


def frac(rs, q):
    xs = [r[f"agree_{q}"]["prefix_frac"] for r in R if r.get(f"agree_{q}")]
    return mean(xs)


def cp(rs, q):
    xs = [r[f"agree_{q}"]["common_prefix"] for r in R if r.get(f"agree_{q}")]
    return mean(xs)


print(f"model={d['model']}  n_prompts={d['n_prompts']}  max_new={d['max_new_tokens']}\n")
print("| 指标 | FP16 参考 | INT4 (AWQ) | INT8 (SmoothQuant) |")
print("|---|---|---|---|")
print(f"| 平均 perplexity(越低越好) | {ppl_fp16:.3f} | {ppl_i4:.3f} | {ppl_i8:.3f} |")
if ppl_fp16:
    print(f"| PPL 相对 FP16 增幅 | — | {(ppl_i4/ppl_fp16-1)*100:+.1f}% | {(ppl_i8/ppl_fp16-1)*100:+.1f}% |")
print(f"| 与 FP16 贪心输出 token 一致前缀占比 | 100% | {frac(R,'int4')*100:.1f}% | {frac(R,'int8')*100:.1f}% |")
print(f"| 平均一致前缀长度(token) | — | {cp(R,'int4'):.1f} | {cp(R,'int8'):.1f} |")

print("\n--- per-prompt ---")
for r in R:
    a4 = r.get("agree_int4") or {}
    a8 = r.get("agree_int8") or {}
    print(f"[{r['idx']:2d}] ppl fp16={r['ppl_fp16_own']:.2f} int4={r['ppl_int4']:.2f} int8={r['ppl_int8']:.2f}"
          f" | prefix int4={a4.get('common_prefix','?')}/{a4.get('ref_len','?')} int8={a8.get('common_prefix','?')}/{a8.get('ref_len','?')}")

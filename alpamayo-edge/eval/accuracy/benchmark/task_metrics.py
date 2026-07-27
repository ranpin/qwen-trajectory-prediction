#!/usr/bin/env python3
"""Task-matched metrics for the Cosmos-Reason1-Benchmark run (robovqa + robofail, n=210).

Why this exists on top of score_mc.py: every one of the 210 items is a **binary yes/no**
question, so plain accuracy is the wrong headline metric.

  * chance is 50%, not 0% -- 76.2% means 26.2 pts of above-chance signal, and a 2.4 pts
    accuracy drop is a 9.1% loss of that signal.
  * a binary decision task needs sensitivity / specificity / balanced accuracy / kappa.
    Accuracy alone hides a *threshold shift*, which is exactly what quantization does here.
  * yes/no is randomized against A/B per item, so semantic bias (always answer "no") and
    positional bias (always answer "A") are separable. They must be reported separately.
  * n=210 with ~12% discordance can only resolve ~6.7 pts; the minimum detectable effect
    has to be stated or "not significant" will be misread as "no difference".

Usage:
    .venv/bin/python alpamayo-edge/eval/accuracy/benchmark/task_metrics.py --out TASK_METRICS.json
"""
import argparse
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
Z_A, Z_B = 1.959963985, 0.8416212336  # two-sided 95%, 80% power

RUNS = [("fp16", None), ("int8", "orin_int8_outputs_n210.json"),
        ("int4", "orin_int4_outputs_n210.json"), ("2b", "orin_2b_outputs.json")]


def parse_choice(text, options):
    t = (text or "").strip()
    m = re.match(r"^\s*\(?([A-Z])\)?\s*[).:\-]?\s*$", t)
    if m and m.group(1) in options:
        return m.group(1)
    m = re.search(r"\b([A-Z])\b", t)
    return m.group(1) if m and m.group(1) in options else None


def classify(q):
    """Two orthogonal axes, assigned from the question template (not guessed).

    modality:    prospective  = asks whether something *can/will* happen
                 retrospective = asks whether something *did* happen
    granularity: instruction  = about the whole instruction
                 subtask      = about one named subtask
    """
    prospective = bool(re.search(r"Is it possible|Will the robot", q))
    subtask = "subtask" in q
    return ("prospective" if prospective else "retrospective",
            "subtask" if subtask else "instruction")


def wilson(k, n, z=Z_A):
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="cosmos_bench210_manifest.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = json.load(open(os.path.join(HERE, a.manifest)))["manifest"]
    n = len(man)
    # ground truth in *semantic* space ("yes"/"no"), not letter space
    truth = [(e["options"][e["answer"]]).strip().lower() for e in man]
    cells = [classify(e["question"]) for e in man]
    chance = 100.0 * max(sum(1 for x in truth if x == "yes"),
                         sum(1 for x in truth if x == "no")) / n

    out = {"n": n, "all_items_binary": all(len(e.get("options") or {}) == 2 for e in man),
           "answer_balance": {"A": sum(1 for e in man if e["answer"] == "A"),
                              "B": sum(1 for e in man if e["answer"] == "B")},
           "truth_yes_pct": 100 * sum(1 for x in truth if x == "yes") / n,
           "chance_baseline_pct": 50.0,
           "majority_class_baseline_pct": chance,
           "task_cells": {}, "runs": {}}

    for cell in sorted(set(cells)):
        idx = [i for i in range(n) if cells[i] == cell]
        out["task_cells"]["/".join(cell)] = len(idx)

    picks = {}
    for tag, f in RUNS:
        if f is None:
            picks[tag] = [e["fp16_pred"] for e in man]
        else:
            d = json.load(open(os.path.join(HERE, f)))
            r = {x["request_idx"]: x.get("output_text", "") for x in d["responses"]}
            picks[tag] = [parse_choice(r[i], man[i].get("options") or {}) for i in range(n)]

    ref = [picks["fp16"][i] == man[i]["answer"] for i in range(n)]
    ref_acc = 100 * sum(ref) / n

    hdr = (f"{'run':6}{'acc%':>7}{'CI':>16}{'balAcc%':>9}{'TPR':>7}{'TNR':>7}"
           f"{'kappa':>7}{'answered-yes%':>14}{'chose-A%':>10}{'aboveChance':>12}{'signalLoss%':>12}")
    print(f"n={n}  全部二选一={out['all_items_binary']}  答案平衡={out['answer_balance']}  "
          f"随机基线=50.0%  多数类基线={chance:.1f}%\n")
    print(hdr)
    for tag, _ in RUNS:
        p = picks[tag]
        corr = [p[i] == man[i]["answer"] for i in range(n)]
        sem = [(man[i]["options"].get(p[i]) or "").strip().lower() for i in range(n)]
        k = sum(corr)
        acc = 100 * k / n
        tp = sum(1 for i in range(n) if truth[i] == "yes" and sem[i] == "yes")
        fn = sum(1 for i in range(n) if truth[i] == "yes" and sem[i] != "yes")
        tn = sum(1 for i in range(n) if truth[i] == "no" and sem[i] == "no")
        fp_ = sum(1 for i in range(n) if truth[i] == "no" and sem[i] != "no")
        tpr, tnr = tp / (tp + fn), tn / (tn + fp_)
        bacc = 100 * (tpr + tnr) / 2
        po = k / n
        pe = ((tp + fp_) * (tp + fn) + (tn + fn) * (tn + fp_)) / n ** 2
        kappa = (po - pe) / (1 - pe)
        yes = 100 * sum(1 for s in sem if s == "yes") / n
        chA = 100 * sum(1 for x in p if x == "A") / n
        above = acc - 50.0
        loss = 100 * (1 - above / (ref_acc - 50.0)) if tag != "fp16" else 0.0
        lo, hi = wilson(k, n)
        print(f"{tag:6}{acc:7.1f}{f'[{lo:.1f},{hi:.1f}]':>16}{bacc:9.1f}{tpr:7.3f}{tnr:7.3f}"
              f"{kappa:7.3f}{yes:14.1f}{chA:10.1f}{above:12.1f}{loss:12.1f}")

        rec = {"acc_pct": acc, "wilson95": [lo, hi], "balanced_acc_pct": bacc,
               "tpr_sensitivity": tpr, "tnr_specificity": tnr, "cohen_kappa": kappa,
               "answered_yes_pct": yes, "chose_A_pct": chA,
               "above_chance_pts": above, "above_chance_signal_loss_pct": loss,
               "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp_}, "per_cell": {}}
        if tag != "fp16":
            b = sum(1 for i in range(n) if ref[i] and not corr[i])
            c = sum(1 for i in range(n) if corr[i] and not ref[i])
            pd = (b + c) / n
            rec["mcnemar"] = {"b_fp16_right_quant_wrong": b, "c_reverse": c,
                              "discordance_pct": 100 * pd,
                              "p_exact": mcnemar_exact(b, c)}
            rec["mde_pts_at_80pct_power"] = 100 * (Z_A + Z_B) * math.sqrt(pd / n)
            delta = abs(acc - ref_acc) / 100
            rec["n_needed_for_80pct_power"] = (
                ((Z_A + Z_B) ** 2 * pd / delta ** 2) if delta > 0 else None)
        for cell in sorted(set(cells)):
            idx = [i for i in range(n) if cells[i] == cell]
            ks = sum(corr[i] for i in idx)
            rec["per_cell"]["/".join(cell)] = {"acc_pct": 100 * ks / len(idx), "n": len(idx)}
        out["runs"][tag] = rec

    print("\n任务分格正确率（modality / granularity）")
    keys = sorted(out["task_cells"])
    labels = [f"{k} (n={out['task_cells'][k]})" for k in keys]
    w = max(len(x) for x in labels) + 2
    print(f"{'run':6}" + "".join(f"{x:>{w}}" for x in labels))
    for tag, _ in RUNS:
        print(f"{tag:6}" + "".join(
            f"{out['runs'][tag]['per_cell'][k]['acc_pct']:>{w}.1f}" for k in keys))

    print("\n统计功效（配对 McNemar）")
    for tag, _ in RUNS:
        r = out["runs"][tag]
        if "mde_pts_at_80pct_power" in r:
            nn = r["n_needed_for_80pct_power"]
            print(f"  {tag:5} 观测差 {r['acc_pct']-ref_acc:+.1f} pts, 不一致率 {r['mcnemar']['discordance_pct']:.1f}%, "
                  f"p={r['mcnemar']['p_exact']:.3f}, **MDE={r['mde_pts_at_80pct_power']:.1f} pts**, "
                  f"要显著需 n≈{nn:,.0f}" if nn else "")

    if a.out:
        json.dump(out, open(os.path.join(HERE, a.out), "w"), indent=1, ensure_ascii=False)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()

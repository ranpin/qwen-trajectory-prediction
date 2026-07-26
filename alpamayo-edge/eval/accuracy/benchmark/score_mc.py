#!/usr/bin/env python3
"""Score multiple-choice runs on Cosmos-Reason1-Benchmark (robovqa + robofail, n=210).

Until now the MC numbers in RESULTS.json had no committed scorer -- the ground truth for
all 210 items lived only on the Orin. Both are now in git:
    cosmos_bench210_manifest.json   210 items with `answer` and the FP16 prediction
    orin_<tag>_outputs.json         llm_inference --outputFile dumps (request order preserved)

Usage:
    .venv/bin/python alpamayo-edge/eval/accuracy/benchmark/score_mc.py \
        --manifest cosmos_bench210_manifest.json \
        --runs int4=orin_int4_outputs.json int8=orin_int8_outputs.json 2b=orin_2b_outputs.json

Reports per-subset and overall accuracy, Wilson 95% CI, and McNemar vs the FP16 reference.
The Wilson helper reproduces the intervals already stored in RESULTS.json exactly, which is
how we know the implementation is right (see eval/README notes).
"""
import argparse
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.959963985):
    """Wilson score interval, returned in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (centre - half), 100 * (centre + half))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def parse_choice(text, options):
    """Extract the chosen option letter from a generation.

    The prompt asks for a single letter, but be tolerant: accept a bare letter, a letter with
    punctuation, or the full option text. Return None if nothing matches so unparseable
    answers are counted as wrong *and* reported, never silently dropped.
    """
    if not text:
        return None
    t = text.strip()
    m = re.match(r"^\s*\(?([A-Z])\)?\s*[).:\-]?\s*$", t)
    if m and m.group(1) in options:
        return m.group(1)
    m = re.match(r"^\s*\(?([A-Z])\)?\s*[).:\-]\s+", t)
    if m and m.group(1) in options:
        return m.group(1)
    for letter, body in options.items():
        if body and t.lower() == str(body).strip().lower():
            return letter
    m = re.search(r"\b([A-Z])\b", t)
    if m and m.group(1) in options:
        return m.group(1)
    return None


def load_run(path):
    d = json.load(open(path))
    resp = d.get("responses") or []
    out = {}
    for r in resp:
        idx = r.get("request_idx", r.get("batch_idx"))
        out[idx] = r.get("output_text", "")
    return [out[i] for i in sorted(out)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="cosmos_bench210_manifest.json")
    ap.add_argument("--runs", nargs="+", required=True, help="tag=file.json ...")
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    a = ap.parse_args()

    man = json.load(open(os.path.join(HERE, a.manifest)))["manifest"]
    subsets = sorted({e["subset"] for e in man})

    fp16_correct = [e.get("fp16_pred") == e["answer"] for e in man]
    results = {"n": len(man), "subsets": {}, "runs": {}}
    print(f"manifest: {len(man)} items, subsets {subsets}\n")

    fp16_k = sum(fp16_correct)
    lo, hi = wilson(fp16_k, len(man))
    print(f"{'FP16 (reference)':22} overall {100*fp16_k/len(man):5.1f}%  "
          f"CI [{lo:.1f},{hi:.1f}]")
    for s in subsets:
        idx = [i for i, e in enumerate(man) if e["subset"] == s]
        k = sum(fp16_correct[i] for i in idx)
        print(f"    {s:12} {100*k/len(idx):5.1f}%  (n={len(idx)})")
    results["runs"]["fp16"] = {"overall": 100 * fp16_k / len(man),
                               "ci": list(wilson(fp16_k, len(man)))}

    for spec in a.runs:
        tag, path = spec.split("=", 1)
        gens = load_run(os.path.join(HERE, path))
        if len(gens) != len(man):
            print(f"\n!! {tag}: {len(gens)} responses vs {len(man)} manifest items -- skipping")
            continue
        correct, unparsed = [], 0
        for e, g in zip(man, gens):
            ch = parse_choice(g, e.get("options") or {})
            if ch is None:
                unparsed += 1
            correct.append(ch == e["answer"])
        k = sum(correct)
        lo, hi = wilson(k, len(man))
        b = sum(1 for f, q in zip(fp16_correct, correct) if f and not q)
        c = sum(1 for f, q in zip(fp16_correct, correct) if q and not f)
        p = mcnemar_exact(b, c)
        print(f"\n{tag:22} overall {100*k/len(man):5.1f}%  CI [{lo:.1f},{hi:.1f}]  "
              f"vs FP16: {100*(k-fp16_k)/len(man):+.1f} pts, McNemar p={p:.3f} "
              f"({'significant' if p < 0.05 else 'NOT significant'})")
        if unparsed:
            print(f"    note: {unparsed} generations could not be parsed to a letter "
                  f"(counted as wrong)")
        per = {}
        for s in subsets:
            idx = [i for i, e in enumerate(man) if e["subset"] == s]
            ks = sum(correct[i] for i in idx)
            per[s] = {"acc": 100 * ks / len(idx), "n": len(idx)}
            print(f"    {s:12} {100*ks/len(idx):5.1f}%  (n={len(idx)})")
        results["runs"][tag] = {"overall": 100 * k / len(man), "ci": [lo, hi],
                                "mcnemar_p_vs_fp16": p, "discordant_b_c": [b, c],
                                "unparsed": unparsed, "per_subset": per}

    if a.out:
        json.dump(results, open(os.path.join(HERE, a.out), "w"), indent=1)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()

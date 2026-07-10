#!/usr/bin/env python3
"""M4 — Benchmark the deployed Alpamayo engine on Jetson Orin.

Measures per-prediction latency + throughput, and (optionally) power/mem via
`tegrastats`. Precision-agnostic: run once per engine (fp16/int8/int4) to build
the accuracy-latency-power tradeoff table.

This is a skeleton — wire the actual inference call at M4 once the engine runs.
"""

import argparse
import json
import subprocess
import time


def run_tegrastats(seconds):
    """Sample tegrastats for power/mem; returns raw lines (parse offline)."""
    try:
        p = subprocess.Popen(["tegrastats", "--interval", "1000"],
                             stdout=subprocess.PIPE, text=True)
        lines = []
        t0 = time.time()
        while time.time() - t0 < seconds:
            lines.append(p.stdout.readline().strip())
        p.terminate()
        return lines
    except FileNotFoundError:
        return ["tegrastats not found (run on Orin)"]


def predict_once(sample):
    """TODO(M4): call the deployed engine; return predicted (N,2) waypoints."""
    raise NotImplementedError("wire TensorRT-Edge-LLM inference here")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--power", action="store_true", help="sample tegrastats")
    ap.add_argument("--output", "-o", default="outputs/bench.json")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.samples)][:args.runs + args.warmup]
    lat = []
    for i, s in enumerate(samples):
        t0 = time.time()
        predict_once(s)            # NotImplementedError until M4
        dt = time.time() - t0
        if i >= args.warmup:
            lat.append(dt)

    import numpy as np
    res = {"n": len(lat), "latency_ms_mean": float(np.mean(lat) * 1000),
           "latency_ms_p50": float(np.percentile(lat, 50) * 1000),
           "latency_ms_p95": float(np.percentile(lat, 95) * 1000),
           "throughput_hz": float(1.0 / np.mean(lat))}
    print(json.dumps(res, indent=2))
    json.dump(res, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()

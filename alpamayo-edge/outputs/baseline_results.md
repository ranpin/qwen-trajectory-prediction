# Baseline results (synthetic AV, placeholder)

Demonstrates the eval harness end-to-end **before** real data/model are available.
Once Alpamayo-R1 (FP16) and Cosmos-Reason2 (INT4/INT8) run on the Orin, the LLM
rows get filled and this synthetic set is replaced by the NGSIM / PhysicalAI-AV subset.

## CVM baseline — synthetic AV (200 samples, 64-waypoint / 6.4s horizon)
| model | ADE (m) | FDE (m) | Miss Rate (>2m) |
|-------|---------|---------|-----------------|
| CVM (constant velocity) | 21.97 mean / **3.93 median** | 42.09 mean / 9.48 median | 87% |
| Alpamayo-R1 FP16 | 🔜 (Orin) | 🔜 | 🔜 |
| Cosmos-Reason2 INT4/INT8 | 🔜 (Orin) | 🔜 | 🔜 |

> Mean ≫ median because synthetic maneuvers include high-speed turns/lane-changes
> that constant-velocity cannot extrapolate (long-tail outliers). Reported as
> single-prediction ADE/FDE (not best-of-K).

Figures: `figures/sample_{0,1,2}.png` (observed=blue, CVM=green, GT=black).

Reproduce:
```
python data/make_synthetic_av.py --n 200 --out data/av_subset/synthetic_test.jsonl
python eval/evaluate.py --samples data/av_subset/synthetic_test.jsonl --baseline cvm \
    --output outputs/eval_cvm_synthetic.json --label "CVM / synthetic-AV (200)"
```

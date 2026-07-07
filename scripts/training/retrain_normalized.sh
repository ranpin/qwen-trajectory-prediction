#!/bin/bash
# Retrain the trajectory model on AGENT-CENTRIC NORMALIZED data, then quantize,
# deploy, and (optionally) evaluate. This is the A3 step of the plan: an A/B test
# of whether coordinate normalization lets the LLM beat the CVM baseline on real
# data (see scripts/evaluation/baselines.py for the baseline it must beat).
#
# RUN THIS ON pc-3070 (Linux + CUDA + RTX 3070), not on the Mac dev box.
#
# Usage:  bash scripts/training/retrain_normalized.sh [normalize_mode]
#   normalize_mode: none | translate | translate_rotate   (default: translate_rotate)
#
# It reuses the existing per-step scripts; edit those for hyperparameters.
set -e

NORMALIZE="${1:-translate_rotate}"
N_SAMPLES="${N_SAMPLES:-50000}"
GGUF_OUT="outputs/qwen3-4b-q4_k_m.gguf"
ORIN_HOST="${ORIN_HOST:-orin}"

echo "=== A3 retrain | normalize=$NORMALIZE | samples=$N_SAMPLES ==="

# 1) Generate normalized synthetic training data (same 50k recipe, normalized).
python scripts/data_prep/synthetic_gen.py \
    --num_samples "$N_SAMPLES" --normalize "$NORMALIZE" \
    --output data/processed/synthetic_sft.jsonl

# 2) QLoRA train + merge (train_synthetic.sh reads synthetic_sft.jsonl).
bash scripts/training/train_synthetic.sh

# 3) GGUF Q4_K_M quantize of the merged model.
#    ms-swift writes the merged model to <checkpoint>-merged (ignoring the
#    trainer's --output_dir), so locate it there rather than assuming a name.
MERGED=$(ls -d outputs/qwen3-4b-synthetic-lora/v*/checkpoint-500-merged 2>/dev/null | tail -1)
[ -z "$MERGED" ] && { echo "merged model not found"; exit 1; }
python scripts/training/quantize_gguf.py \
    --model_path "$MERGED" \
    --output_path "$GGUF_OUT" --quant_type Q4_K_M

# 4) Deploy to Orin and (re)start the server.
bash scripts/deploy/deploy_to_orin.sh "$GGUF_OUT" "$ORIN_HOST"

cat <<NEXT

=== deployed. Now evaluate (normalization is metric-invariant, so CVM numbers
    are unchanged; only the LLM's numbers move):

  # build the normalized test set + the SAME stratified 150-sample subset
  python scripts/data_prep/preprocess.py --normalize $NORMALIZE
  python scripts/evaluation/sample_testset.py \\
      --test_file data/processed/trajectory_test.jsonl \\
      --output_file data/processed/eth_ucy_test_sample_norm.jsonl

  # generate predictions from the NORMALIZED model, then score
  set -a && . configs/deploy.env && set +a
  python scripts/evaluation/generate_predictions.py \\
      --test_file data/processed/eth_ucy_test_sample_norm.jsonl \\
      --output_file data/processed/eth_ucy_predictions_norm.jsonl --max_samples 150
  python scripts/evaluation/evaluate.py \\
      --test_file data/processed/eth_ucy_test_sample_norm.jsonl \\
      --predictions_file data/processed/eth_ucy_predictions_norm.jsonl \\
      --output_file outputs/eval_results_eth_ucy_norm.json \\
      --label "ETH/UCY $NORMALIZE"

  # run the demo against the normalized model
  python demo/app.py --normalize $NORMALIZE
NEXT

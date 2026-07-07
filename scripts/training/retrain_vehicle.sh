#!/bin/bash
# Train a VEHICLE-primary trajectory model (keeps pedestrian via mixed data) on
# agent-centric normalized synthetic data, then quantize to GGUF. Self-contained
# (does not call train_synthetic.sh) and writes to vehicle-specific output dirs
# so the pedestrian model is never overwritten.
#
# RUN ON pc-3070 (Linux + CUDA + RTX 3070). Deploy is done from the Mac.
#
# Usage:  bash scripts/training/retrain_vehicle.sh [normalize_mode]
#   normalize_mode: none | translate | translate_rotate   (default translate_rotate)
# Env:  AGENT=mixed|vehicle (default mixed), VEHICLE_RATIO=0.75, N_SAMPLES=50000
set -e
cd "$(dirname "$0")/../.."

NORMALIZE="${1:-translate_rotate}"
AGENT="${AGENT:-mixed}"
VEHICLE_RATIO="${VEHICLE_RATIO:-0.75}"
N_SAMPLES="${N_SAMPLES:-50000}"
MODEL_PATH="models/qwen/Qwen3-4B"
DATA_FILE="data/processed/vehicle_sft.jsonl"
OUTPUT_DIR="outputs/qwen3-4b-vehicle-lora"
GGUF_OUT="outputs/qwen3-4b-vehicle-q4_k_m.gguf"

echo "=== vehicle retrain | agent=$AGENT ratio=$VEHICLE_RATIO normalize=$NORMALIZE ==="
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); \
  print('GPU:', torch.cuda.get_device_name(0))"

echo "[veh] $(date '+%F %T') gen synthetic vehicle data"
python scripts/data_prep/synthetic_gen.py \
    --num_samples "$N_SAMPLES" --agent_type "$AGENT" \
    --vehicle_ratio "$VEHICLE_RATIO" --normalize "$NORMALIZE" \
    --output "$DATA_FILE"

echo "[veh] $(date '+%F %T') QLoRA train"
swift sft \
    --model "$MODEL_PATH" --tuner_type lora --dataset "$DATA_FILE" \
    --learning_rate 1e-4 --lora_rank 16 --lora_alpha 32 \
    --target_modules all-linear --max_steps 500 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --max_length 1024 --quant_method bnb --quant_bits 4 \
    --torch_dtype bfloat16 --output_dir "$OUTPUT_DIR" \
    --save_steps 500 --logging_steps 10 --gradient_checkpointing true \
    --optim paged_adamw_32bit --warmup_ratio 0.05

echo "[veh] $(date '+%F %T') merge LoRA"
CKPT=$(ls -d ${OUTPUT_DIR}/v*/checkpoint-500 2>/dev/null | tail -1)
[ -z "$CKPT" ] && { echo "no checkpoint found"; exit 1; }
CUDA_VISIBLE_DEVICES='' swift export \
    --model "$MODEL_PATH" --adapters "$CKPT" --merge_lora true
# ms-swift writes the merged model to <adapters>-merged, ignoring --output_dir.
MERGED="${CKPT}-merged"
[ -d "$MERGED" ] || { echo "merged model not found at $MERGED"; exit 1; }

echo "[veh] $(date '+%F %T') GGUF Q4_K_M quantize (from $MERGED)"
python scripts/training/quantize_gguf.py \
    --model_path "$MERGED" \
    --output_path "$GGUF_OUT" --quant_type Q4_K_M

echo "[veh] $(date '+%F %T') DONE_ALL -> $GGUF_OUT"
echo "Next (from Mac): scp gguf back, deploy_to_orin.sh <gguf> orin-dog,"
echo "  then eval synthetic-vehicle + NGSIM with --agent_type vehicle."

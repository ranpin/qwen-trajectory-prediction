#!/bin/bash
# Train Qwen3-4B with QLoRA using synthetic data
# This script uses the pre-generated 50000 synthetic samples

set -e

PROJECT_DIR="/data/tmp/chenrunbin/projects/qwen-trajectory-prediction"
cd "$PROJECT_DIR"

# Activate venv
source .venv/bin/activate

# Check GPU
echo "Checking GPU..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
python3 -c "import torch; vram = torch.cuda.get_device_properties(0).total_memory / 1024**3; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.1f}GB')"

# Paths
DATA_FILE="data/processed/synthetic_sft.jsonl"
OUTPUT_DIR="outputs/qwen3-4b-synthetic-lora"
MODEL_PATH="models/qwen/Qwen3-4B"

# Check data
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: $DATA_FILE not found!"
    echo "Run synthetic data generation first: python scripts/data_prep/synthetic_gen.py"
    exit 1
fi

# Check model
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH!"
    echo "Download from ModelScope first:"
    echo "  python -c 'from modelscope import snapshot_download; snapshot_download(\"qwen/Qwen3-4B\", cache_dir=\"./models\")'"
    exit 1
fi

NUM_SAMPLES=$(wc -l < "$DATA_FILE")
echo "Training data: $DATA_FILE ($NUM_SAMPLES samples)"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Train with ms-swift using synthetic data
echo ""
echo "Starting QLoRA training with synthetic data..."
echo ""

swift sft \
    --model "$MODEL_PATH" \
    --tuner_type lora \
    --dataset "$DATA_FILE" \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --max_steps 500 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 1024 \
    --quant_method bnb \
    --quant_bits 4 \
    --torch_dtype bfloat16 \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 500 \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --optim paged_adamw_32bit \
    --warmup_ratio 0.05

echo ""
echo "Training completed! LoRA adapter saved to $OUTPUT_DIR"
echo ""

# Merge LoRA weights (ms-swift 4.x syntax)
echo "Merging LoRA weights..."
CHECKPOINT_DIR=$(ls -d ${OUTPUT_DIR}/v*/checkpoint-500 2>/dev/null | head -1)
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: No checkpoint found in $OUTPUT_DIR"
    exit 1
fi
CUDA_VISIBLE_DEVICES='' swift export \
    --model "$MODEL_PATH" \
    --adapters "$CHECKPOINT_DIR" \
    --merge_lora true \
    --output_dir "${OUTPUT_DIR}-merged"

echo ""
echo "Merged model saved to ${OUTPUT_DIR}-merged"
echo ""
echo "Next steps:"
echo "  1. AWQ quantize: CUDA_VISIBLE_DEVICES='' swift export --model $MODEL_PATH --adapters $CHECKPOINT_DIR --merge_lora true --quant_method awq --quant_bits 4 --quant_n_samples 256 --dataset $DATA_FILE --output_dir outputs/qwen3-4b-awq"
echo "  2. GGUF quantize: python scripts/training/quantize_gguf.py --model_path ${OUTPUT_DIR}-merged"
echo "  3. Deploy to Orin: bash scripts/deploy/deploy_to_orin.sh"

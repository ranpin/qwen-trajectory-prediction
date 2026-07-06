#!/bin/bash
# Train Qwen3-4B with QLoRA for trajectory prediction
# Requires: 8GB VRAM GPU (RTX 3070)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

# Activate venv
source .venv/bin/activate

# Check GPU
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
python3 -c "import torch; vram = torch.cuda.get_device_properties(0).total_memory / 1024**3; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.1f}GB')"

# Paths
DATA_FILE="data/processed/trajectory_sft.json"
OUTPUT_DIR="outputs/qwen3-4b-trajectory-lora"

# Check data
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: $DATA_FILE not found!"
    echo "Run preprocessing first: python scripts/data_prep/preprocess.py"
    exit 1
fi

echo "Starting QLoRA training..."
echo "Data: $DATA_FILE"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Train with ms-swift
swift sft \
    --model Qwen/Qwen3-4B \
    --tuner_type lora \
    --dataset "$DATA_FILE" \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --quantization_bit 4 \
    --torch_dtype bfloat16 \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --optim paged_adamw_32bit

echo "Training completed! Model saved to $OUTPUT_DIR"

# Merge LoRA weights
echo "Merging LoRA weights..."
swift export \
    --model "$OUTPUT_DIR" \
    --merge_lora true \
    --output_dir "${OUTPUT_DIR}-merged"

echo "Merged model saved to ${OUTPUT_DIR}-merged"

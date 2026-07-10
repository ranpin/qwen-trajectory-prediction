#!/bin/bash
# INT4/INT8 quantization track — TensorRT-Edge-LLM QUANTIZATION applies to
# SUPPORTED LLM/VLM checkpoints, NOT Alpamayo (Alpamayo is FP16-only in v0.9.0).
# Recommended target: nvidia/Cosmos-Reason2-8B (same Physical-AI family, VLM) or a
# Qwen3-VL. Runs on an x86 host with an NVIDIA GPU. Ref: features/quantization.md
set -e
QMODEL="${QMODEL:-nvidia/Cosmos-Reason2-8B}"
QFORMAT="${QFORMAT:-int4_awq}"        # Orin runtime: int4_awq | int8_sq (NO fp8/nvfp4)
OUT="${OUT:-checkpoints/${QMODEL##*/}-${QFORMAT}}"

tensorrt-edgellm-quantize llm \
    --model_dir "$QMODEL" \
    --output_dir "$OUT" \
    --qformat "$QFORMAT"
    # VLM visual tower stays FP16 on Orin (fp8 visual needs Blackwell/Thor)

echo "quantized checkpoint -> $OUT"
echo "next: tensorrt-edgellm-export '$OUT' '$OUT/onnx' → scp to device → *_build → run"

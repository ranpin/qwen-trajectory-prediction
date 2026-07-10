#!/bin/bash
# M3 — Quantize Alpamayo-R1 with TensorRT-Edge-LLM (runs on the Orin, 64GB).
# Orin runtime supports FP16 / INT8 / INT4 only (NOT FP8/FP4/NVFP4 — Blackwell).
# Ref: NVIDIA/TensorRT-Edge-LLM docs (features/quantization.md, supported-models.md)
set -e

MODEL="${MODEL:-nvidia/Alpamayo-R1-10B}"          # HF checkpoint (BF16)
QFORMAT="${QFORMAT:-int4_awq}"                     # int4_awq | int8_sq
OUT="${OUT:-checkpoints/alpamayo-r1-${QFORMAT}}"
CALIB="${CALIB:-data/av_subset/calib.jsonl}"       # small calibration set

# TODO(M0): pin exact CLI/flags from TensorRT-Edge-LLM Quick Start.
# Documented tools: tensorrt-edgellm-quantize  /  tensorrt-edgellm-export
tensorrt-edgellm-quantize llm \
    --model "$MODEL" \
    --qformat "$QFORMAT" \
    --calib "$CALIB" \
    --output "$OUT"
    # visual encoder stays FP16 on Orin (do NOT use --visual_quantization fp8 here)

echo "quantized checkpoint -> $OUT"

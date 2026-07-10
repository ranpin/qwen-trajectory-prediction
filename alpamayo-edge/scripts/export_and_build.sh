#!/bin/bash
# M3/M4 — Export quantized checkpoint and build the edge engine ON the Orin.
# TensorRT-Edge-LLM workflow: HF/quantized ckpt --(export)--> ONNX --(build)--> engine,
# with engine build + inference running entirely on the edge device.
set -e

CKPT="${CKPT:-checkpoints/alpamayo-r1-int4_awq}"
ENGINE_DIR="${ENGINE_DIR:-engines/alpamayo-r1-int4}"
PREC="${PREC:-int4}"                                # fp16 | int8 | int4 (Orin)

# TODO(M0): pin exact flags from Quick Start / checkpoint-export.md.
tensorrt-edgellm-export \
    --checkpoint "$CKPT" \
    --output "$ENGINE_DIR" \
    --precision "$PREC"
    # For Orin Nano (low RAM) add: --externalize-weights int4_ffn
    # Alpamayo action head (flow-matching diffusion) is exported via the
    # tensorrt_edgellm/models/alpamayo module + cpp/action/alpamayo1ActionRunner.

echo "engine -> $ENGINE_DIR"

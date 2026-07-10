#!/bin/bash
# Alpamayo-R1 FP16 workflow (official TensorRT-Edge-LLM examples/vla.md).
# NOTE: Alpamayo is FP16-ONLY in v0.9.0 — there is NO INT4/INT8 for the VLA yet.
# Export runs on an x86 host w/ GPU; engines build on the edge device
# (official example uses Jetson THOR; Orin FP16 plausible but unverified).
set -e
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/tensorrt-edgellm-workspace}"
MODEL_NAME="${MODEL_NAME:-Alpamayo-R1-10B}"
KV="${KV:-4096}"
DEVICE="${ORIN_HOST:-vision@30.245.40.99}"
W="$WORKSPACE_DIR/$MODEL_NAME"

echo "### Step 1 (x86 HOST): download + export to ONNX (FP16)"
# hf auth login
# hf download nvidia/${MODEL_NAME} --local-dir "$W"
tensorrt-edgellm-export "$W" "$W/onnx" --max-kv-cache-capacity "$KV"   # -> onnx/{llm,visual,action}

echo "### Step 2: transfer ONNX to edge device"
scp -r "$W/onnx" "$DEVICE:~/tensorrt-edgellm-workspace/$MODEL_NAME/"

cat <<REMOTE

### Step 3 (RUN ON DEVICE, inside the TensorRT-Edge-LLM repo):
W=~/tensorrt-edgellm-workspace/$MODEL_NAME
./build/examples/llm/llm_build          --onnxDir \$W/onnx/llm    --engineDir \$W/engines/llm --maxInputLen 3424 --maxKVCacheCapacity $KV --maxBatchSize 6
./build/examples/multimodal/visual_build --onnxDir \$W/onnx/visual --engineDir \$W/engines    --minImageTokens 160 --maxImageTokens 18432 --maxImageTokensPerImage 192
./build/examples/multimodal/action_build --onnxDir \$W/onnx/action --engineDir \$W/engines    --maxBatchSize 6
# then: scripts/run_action_inference.sh
REMOTE

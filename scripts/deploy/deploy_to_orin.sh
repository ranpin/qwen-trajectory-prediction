#!/bin/bash
# Deploy a quantized GGUF to the Jetson Orin: replace the served model file and
# restart the EXISTING llama.cpp server. The Orin already has a CMake-built
# llama.cpp (build/bin/llama-server) and a fixed served-model path — we do NOT
# clone or rebuild llama.cpp (the old Makefile build no longer exists upstream).
#
# Usage:  bash scripts/deploy/deploy_to_orin.sh <model.gguf> [orin_host]
#   orin_host default: $ORIN_HOST from configs/deploy.env, else vision@30.245.40.99
# Orin-side paths are overridable via env (ORIN_MODEL_DIR / ORIN_LLAMA_CPP / ...).
set -e

MODEL_PATH="${1:?Usage: $0 <model.gguf> [orin_host]}"
ORIN_HOST="${2:-${ORIN_HOST:-vision@30.245.40.99}}"
ORIN_MODEL_DIR="${ORIN_MODEL_DIR:-/home/vision/chenrunbin/qwen-trajectory-prediction}"
ORIN_LLAMA_CPP="${ORIN_LLAMA_CPP:-/home/vision/chenrunbin/llama.cpp}"
ORIN_MODEL_NAME="${ORIN_MODEL_NAME:-qwen3-4b-q4_k_m.gguf}"
PORT="${ORIN_PORT:-8080}"

[ -f "$MODEL_PATH" ] || { echo "Error: model file $MODEL_PATH not found"; exit 1; }
REMOTE_MODEL="$ORIN_MODEL_DIR/$ORIN_MODEL_NAME"
echo "Deploying $(du -h "$MODEL_PATH" | cut -f1) -> $ORIN_HOST:$REMOTE_MODEL"

# Back up the currently-served model once, then upload the new one over it.
ssh "$ORIN_HOST" "[ -f '$REMOTE_MODEL' ] && [ ! -f '$REMOTE_MODEL.bak' ] && cp '$REMOTE_MODEL' '$REMOTE_MODEL.bak' || true"
echo "Uploading model..."
scp "$MODEL_PATH" "$ORIN_HOST:$REMOTE_MODEL"

# Restart the existing server, fully detached (survives ssh disconnect).
echo "Restarting llama.cpp server..."
ssh "$ORIN_HOST" "pkill -f '[l]lama-server' 2>/dev/null; sleep 3; \
  cd '$ORIN_LLAMA_CPP' && PATH=/usr/local/cuda/bin:\$PATH setsid ./build/bin/llama-server \
  --model '$REMOTE_MODEL' --host 0.0.0.0 --port $PORT -c 2048 -ngl 99 -t 8 \
  > '$ORIN_MODEL_DIR/server.log' 2>&1 </dev/null &"

sleep 10
echo "Deployment done. Verify with: curl http://${ORIN_HOST#*@}:$PORT/v1/models"

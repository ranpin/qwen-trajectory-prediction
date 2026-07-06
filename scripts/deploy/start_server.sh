#!/bin/bash
# Start llama-server on Orin with Qwen3-4B trajectory model

MODEL_PATH="/home/vision/chenrunbin/qwen-trajectory-prediction/qwen3-4b-q4_k_m.gguf"
LOG_FILE="/home/vision/chenrunbin/qwen-trajectory-prediction/server.log"
PID_FILE="/home/vision/chenrunbin/qwen-trajectory-prediction/server.pid"
PORT=8080

export PATH=/usr/local/cuda/bin:$PATH
cd /home/vision/chenrunbin/llama.cpp

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Server already running (PID $(cat "$PID_FILE"))"
    exit 0
fi

echo "Starting llama-server on port $PORT..."
nohup ./build/bin/llama-server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --ctx-size 2048 \
    --n-gpu-layers 99 \
    --threads 8 \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 3
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Server started (PID $(cat "$PID_FILE"))"
    echo "API: http://0.0.0.0:$PORT"
    tail -3 "$LOG_FILE"
else
    echo "Failed to start server. Check $LOG_FILE"
    exit 1
fi

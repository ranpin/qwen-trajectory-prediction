#!/bin/bash
# Deploy quantized model to NVIDIA Orin via llama.cpp

set -e

MODEL_PATH="${1:?Usage: $0 <model.gguf> [orin_host]}"
ORIN_HOST="${2:-orin}"
REMOTE_DIR="/data/models/trajectory-prediction"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found!"
    exit 1
fi

MODEL_NAME=$(basename "$MODEL_PATH")
MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)

echo "Deploying model: $MODEL_PATH ($MODEL_SIZE)"
echo "Target: $ORIN_HOST:$REMOTE_DIR"

# Create remote directory
ssh "$ORIN_HOST" "mkdir -p $REMOTE_DIR"

# Copy model
echo "Uploading model..."
scp "$MODEL_PATH" "$ORIN_HOST:$REMOTE_DIR/$MODEL_NAME"

# Copy llama.cpp server (if not already there)
echo "Setting up llama.cpp on Orin..."
ssh "$ORIN_HOST" bash <<'REMOTE_SCRIPT'
if [ ! -d ~/llama.cpp ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
    cd ~/llama.cpp
    make LLAMA_CUBLAS=1 llama-server
fi
REMOTE_SCRIPT

# Create startup script
echo "Creating startup script..."
ssh "$ORIN_HOST" "cat > $REMOTE_DIR/start_server.sh" <<EOF
#!/bin/bash
cd ~/llama.cpp
./llama-server \\
    -m $REMOTE_DIR/$MODEL_NAME \\
    --host 0.0.0.0 \\
    --port 8080 \\
    -ngl 99 \\
    -c 2048 \\
    -t 8
EOF

ssh "$ORIN_HOST" "chmod +x $REMOTE_DIR/start_server.sh"

echo ""
echo "Deployment complete!"
echo ""
echo "To start the server on Orin:"
echo "  ssh $ORIN_HOST"
echo "  $REMOTE_DIR/start_server.sh"
echo ""
echo "To test:"
echo "  curl http://$ORIN_HOST:8080/v1/chat/completions \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"model\":\"qwen3-4b-trajectory\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}]}'"

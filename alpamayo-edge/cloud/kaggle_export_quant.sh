#!/bin/bash
# ONE-TIME heavy step on FREE cloud (Kaggle 2xT4=32GB, or Colab).
# Paste into a notebook cell as:  !bash kaggle_export_quant.sh
# Produces: Alpamayo-R1 FP16 ONNX + Cosmos-Reason2-8B INT4/INT8 checkpoints,
# then zips them for download -> scp to Orin.  (No training; ~one run.)
#
# Prereqs in the notebook first:
#   !pip -q install huggingface_hub && huggingface-cli login   # paste HF token
#   (accept licenses for nvidia/Alpamayo-R1-10B on HF first)
set -e
export DEBIAN_FRONTEND=noninteractive
WORK=${WORK:-/kaggle/working}; cd "$WORK"

# --- toolkit ---
git clone --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM && pip3 -q install ".[tools]"
export EDGE_LLM_PATH=$PWD PYTHONPATH=$PWD:$PYTHONPATH
cd "$WORK"

# --- Track 1: Alpamayo-R1 FP16 export (VLA) ---
hf download nvidia/Alpamayo-R1-10B --local-dir Alpamayo-R1-10B
tensorrt-edgellm-export Alpamayo-R1-10B Alpamayo-R1-10B/onnx --max-kv-cache-capacity 4096
#   -> Alpamayo-R1-10B/onnx/{llm,visual,action}

# --- Track 2: Cosmos-Reason2-8B quantization (INT4 + INT8 for Orin) ---
for QF in int4_awq int8_sq; do
  tensorrt-edgellm-quantize llm \
     --model_dir nvidia/Cosmos-Reason2-8B \
     --output_dir "Cosmos-Reason2-8B-${QF}" --qformat "$QF"
  tensorrt-edgellm-export "Cosmos-Reason2-8B-${QF}" "Cosmos-Reason2-8B-${QF}/onnx"
done

# --- package for download (ONNX only; small vs full weights) ---
tar -czf edge_artifacts.tgz \
  Alpamayo-R1-10B/onnx \
  Cosmos-Reason2-8B-int4_awq/onnx Cosmos-Reason2-8B-int8_sq/onnx
echo "DONE -> $WORK/edge_artifacts.tgz  (download, then scp to Orin ~/tensorrt-edgellm-workspace/)"

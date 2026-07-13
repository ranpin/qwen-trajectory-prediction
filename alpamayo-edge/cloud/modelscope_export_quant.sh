#!/bin/bash
# ONE-TIME heavy step on ModelScope 魔搭 free GPU notebook (国内直连).
# 在 ModelScope Notebook 的 cell 里运行:  !bash modelscope_export_quant.sh
# 产出 Alpamayo-R1 FP16 ONNX + Cosmos-Reason2-8B INT4/INT8,打包到 /mnt/workspace 供下载。
#
# 先决条件(在 notebook 里先跑):
#   !pip -q install -U huggingface_hub
#   import os; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"     # 国内镜像
#   !huggingface-cli login       # 贴 HF token(且已在 HF 网站同意 Alpamayo/Cosmos 许可)
set -e
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}    # 国内下载走镜像
export DEBIAN_FRONTEND=noninteractive
WORK=${WORK:-/mnt/workspace}; cd "$WORK"     # ModelScope 持久目录

echo "### GPU / 内存"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | awk '/Mem:/{print "RAM total", $2"GB"}'

echo "### 安装 TensorRT-Edge-LLM"
[ -d TensorRT-Edge-LLM ] || git clone --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM && pip3 -q install ".[tools]"
export EDGE_LLM_PATH=$PWD PYTHONPATH=$PWD:$PYTHONPATH
cd "$WORK"

echo "### 轨道1: Cosmos-Reason2-8B → INT4/INT8 (较小,先跑;~24GB 可行)"
for QF in int4_awq int8_sq; do
  tensorrt-edgellm-quantize llm --model_dir nvidia/Cosmos-Reason2-8B \
      --output_dir "Cosmos-Reason2-8B-${QF}" --qformat "$QF"
  tensorrt-edgellm-export "Cosmos-Reason2-8B-${QF}" "Cosmos-Reason2-8B-${QF}/onnx"
done

echo "### 轨道2: Alpamayo-R1-10B → FP16 ONNX (显存吃紧,免费实例可能 OOM;挑最大免费 GPU)"
hf download nvidia/Alpamayo-R1-10B --local-dir Alpamayo-R1-10B
tensorrt-edgellm-export Alpamayo-R1-10B Alpamayo-R1-10B/onnx --max-kv-cache-capacity 4096

echo "### 打包(ONNX,几 GB)→ 在 Notebook 文件浏览器右键下载 edge_artifacts.tgz"
tar -czf "$WORK/edge_artifacts.tgz" \
  Alpamayo-R1-10B/onnx \
  Cosmos-Reason2-8B-int4_awq/onnx Cosmos-Reason2-8B-int8_sq/onnx
echo "DONE -> $WORK/edge_artifacts.tgz"

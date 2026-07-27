#!/bin/bash
# 建 maxInputLen=4096 / batch=1 / KV=8192 的 LLM 引擎 + maxImageTokens=4096 的视觉引擎
# 老引擎 (engines/int4, engines/int8) 一律不动，保住已发布数字的可比性
A=/home/vision/chenrunbin/alpamayo-edge
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
export PATH=/usr/local/cuda/bin:$PATH
B=/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm
V=/home/vision/TensorRT-Edge-LLM/build_orin/examples/multimodal
L=$A/logs; mkdir -p "$L"

run_build() {   # tag  cmd...
  tag=$1; shift
  echo "=== BUILD $tag start $(date +%T) ==="
  ( while true; do free -m | awk '/Mem/{print $3}'; sleep 2; done > /tmp/mem_$tag.log ) &
  MON=$!
  "$@" > "$L/build_$tag.log" 2>&1
  rc=$?
  kill $MON 2>/dev/null
  echo "  $tag exit=$rc  build峰值内存=$(sort -n /tmp/mem_$tag.log | tail -1) MB  用时到 $(date +%T)"
}

run_build int4_len4096 $B/llm_build \
  --onnxDir $A/onnx/int4/Cosmos-8B-int4_awq-onnx/llm --engineDir $A/engines/int4_len4096/llm \
  --maxInputLen 4096 --maxBatchSize 1 --maxKVCacheCapacity 8192
du -sh $A/engines/int4_len4096/llm

run_build int8_len4096 $B/llm_build \
  --onnxDir $A/onnx/int8/Cosmos-8B-int8_sq-onnx/llm --engineDir $A/engines/int8_len4096/llm \
  --maxInputLen 4096 --maxBatchSize 1 --maxKVCacheCapacity 8192
du -sh $A/engines/int8_len4096/llm

run_build visual_len4096 $V/visual_build \
  --onnxDir $A/onnx/int4/Cosmos-8B-int4_awq-onnx/visual --engineDir $A/engines/int4_len4096 \
  --maxImageTokens 4096 --maxImageTokensPerImage 2048
du -sh $A/engines/int4_len4096/visual

echo "ALL_BUILDS_DONE $(date +%T)"

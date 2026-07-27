#!/bin/bash
# 目的：int8_len4096 在 512/1024 处慢 15% —— 但它同时改了 3 个变量
#   maxInputLen 1024->4096 (=> TRT opt 形状 512->2048)、maxBatchSize 4->1、maxKVCacheCapacity 4096->8192
# 两个对照各只改一个变量，把 15% 归因到唯一的因上。
A=/home/vision/chenrunbin/alpamayo-edge
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
B=/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm
ONNX=$A/onnx/int8/Cosmos-8B-int8_sq-onnx/llm
L=$A/logs

b(){ tag=$1; shift; echo "=== $tag start $(date +%T) ==="
  ( while true; do free -m|awk '/Mem/{print $3}'; sleep 2; done > /tmp/mc_$tag.log ) & M=$!
  $B/llm_build --onnxDir $ONNX --engineDir $A/engines/$tag/llm "$@" > $L/build_$tag.log 2>&1
  echo "  $tag exit=$? peak=$(sort -n /tmp/mc_$tag.log|tail -1)MB $(date +%T)"; kill $M 2>/dev/null; }

# C1: 只改 batch (4->1)，maxInputLen/KV 与老引擎相同
b int8_ctl_b1   --maxInputLen 1024 --maxBatchSize 1 --maxKVCacheCapacity 4096
# C2: 只改 KV (4096->8192)，maxInputLen/batch 与老引擎相同
b int8_ctl_kv8k --maxInputLen 1024 --maxBatchSize 4 --maxKVCacheCapacity 8192
echo CONTROLS_BUILT

BIN=$B/llm_bench
for t in int8_ctl_b1 int8_ctl_kv8k; do
  for LEN in 512 1024; do
    o=$($BIN --engineDir $A/engines/$t/llm --mode prefill --inputLen $LEN --iterations 10 --warmup 3 2>&1)
    e=$(echo "$o"|grep -oE "E2E Time \(actual performance\): [0-9.]+"|tail -1|grep -oE "[0-9.]+$")
    echo "CTL,$t,prefill,$LEN,$e"
  done
done
echo CONTROLS_DONE $(date +%T)

#!/bin/bash
# 2026-07-27 路径迁移: /home/nvidia/chenrunbin.crb -> /home/nvidia/chenrunbin/alpamayo-edge
#   build_orin -> trtedge/ , e_int* -> onnx/ , *.tgz -> tarballs/ , build 日志 -> logs/
cd /home/nvidia/chenrunbin/alpamayo-edge || exit 1
export EDGELLM_PLUGIN_PATH=/home/nvidia/chenrunbin/alpamayo-edge/trtedge/build_orin/libNvInfer_edgellm_plugin.so
BB=trtedge/build_orin/examples/llm/llm_build
BENCH=trtedge/build_orin/examples/llm/llm_bench
# extract
rm -rf onnx/int4 onnx/int8; mkdir -p onnx/int4 onnx/int8
tar xzf tarballs/edge_artifacts_int4_awq.tgz -C onnx/int4
tar xzf tarballs/edge_artifacts_int8_sq.tgz -C onnx/int8
I4=$(dirname "$(find onnx/int4 -name model.onnx -path '*llm*' | head -1)")
I8=$(dirname "$(find onnx/int8 -name model.onnx -path '*llm*' | head -1)")
echo "I4=$I4  I8=$I8"
# build (same params as fp16 goat build for consistency)
for pair in "int4 $I4" "int8 $I8"; do
  set -- $pair; tag=$1; onnx=$2
  echo "BUILD $tag $(date +%T)"
  $BB --onnxDir "$onnx" --engineDir engines/$tag/llm --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 2048 > logs/build_$tag.log 2>&1
  echo "  build exit=$? engine=$(ls -la engines/$tag/llm/*.engine 2>/dev/null | awk '{printf "%.1fG",$5/1e9}')"
done
# bench all three
echo "===== BENCH (goat, MAXN, batch1) ====="
for tag in fp16 int8 int4; do
  pf=$($BENCH --engineDir engines/$tag/llm --mode prefill --inputLen 512 --iterations 10 --warmup 3 2>&1 | grep -oE "Prefill E2E Time: [0-9.]+" | grep -oE "[0-9.]+$")
  dc=$($BENCH --engineDir engines/$tag/llm --mode decode --pastKVLen 512 --iterations 10 --warmup 3 2>&1 | grep -oE "Decode E2E Time: [0-9.]+" | grep -oE "[0-9.]+$")
  echo "RESULT $tag: TTFT(prefill512)=${pf}ms  TPOT(decode)=${dc}ms"
done
echo ALLDONE

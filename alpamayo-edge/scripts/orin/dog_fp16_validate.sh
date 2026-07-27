#!/bin/bash
# 2026-07-27 路径迁移：两端文件都已归到各自的 chenrunbin/alpamayo-edge/ 下
A=/home/vision/chenrunbin/alpamayo-edge
cd "$A" || exit 1
GOAT=nvidia@30.245.40.73
SRC=/home/nvidia/chenrunbin/alpamayo-edge/tarballs/fp16_engine.tar
DST=$A/tarballs/fp16_engine.tar
WANT=16407797760
for i in $(seq 1 30); do
  cur=$(stat -c%s "$DST" 2>/dev/null || echo 0)
  [ "$cur" = "$WANT" ] && break
  echo "pull attempt $i: have $((cur/1000000))MB / 16408MB"
  rsync --partial --append-verify --timeout=120 -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=15" "$GOAT:$SRC" "$DST" 2>/dev/null
  sleep 2
done
cur=$(stat -c%s "$DST" 2>/dev/null || echo 0)
echo "PULL size=$cur want=$WANT"
[ "$cur" = "$WANT" ] || { echo "INCOMPLETE_ABORT"; exit 1; }
rm -rf "$A"/engines/fp16; mkdir -p "$A"/engines/fp16
tar xf "$DST" -C "$A"/engines/fp16
ls -la "$A"/engines/fp16/llm/*.engine 2>/dev/null | awk '{printf "engine=%.2fGB\n",$5/1e9}'
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
B=/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_bench
( while true; do free -m|awk '/Mem/{print $3}'; sleep 1; done > /tmp/dog_infer_mem.log ) & MON=$!
echo "FP16_PREFILL: $($B --engineDir "$A"/engines/fp16/llm --mode prefill --inputLen 512 --iterations 10 --warmup 3 2>&1 | grep -oE 'Prefill E2E Time: [0-9.]+ .* ms|Prefill E2E Time: [0-9.]+' | head -1)"
echo "FP16_DECODE: $($B --engineDir "$A"/engines/fp16/llm --mode decode --pastKVLen 512 --iterations 10 --warmup 3 2>&1 | grep -oE 'Decode E2E Time: [0-9.]+' | head -1)"
kill $MON 2>/dev/null
echo "PEAK_INFER_MEM_MB=$(sort -n /tmp/dog_infer_mem.log | tail -1)"
echo DOG_FP16_DONE

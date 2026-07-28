#!/bin/bash
# 30 分钟持续 decode 负载 + tegrastats 采温度/频率/功耗
# 动因：已发布的全部延迟数字都是 10~200 次迭代的**突发**负载；车上是连续跑的。
# 问题：TPOT 会不会因降频而退化？
A=/home/vision/chenrunbin/alpamayo-edge
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
BIN=/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_bench
OUT=$A/bench/sustained_30min.csv
TG=$A/logs/sustained_tegrastats.log
echo "run,elapsed_s,e2e_ms,tok_s" > "$OUT"
tegrastats --interval 1000 > "$TG" 2>&1 & TS=$!
T0=$(date +%s)
r=0
while [ $(( $(date +%s) - T0 )) -lt 1800 ]; do
  r=$((r+1))
  o=$("$BIN" --engineDir $A/engines/int4/llm --mode decode --pastKVLen 512 --iterations 2000 --warmup 3 2>&1)
  e=$(echo "$o" | grep -oE "E2E Time \(actual performance\): [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  t=$(echo "$o" | grep -oE "Tokens/sec \(E2E\): [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  el=$(( $(date +%s) - T0 ))
  echo "$r,$el,$e,$t" | tee -a "$OUT"
done
kill $TS 2>/dev/null
echo "SUSTAINED_DONE runs=$r elapsed=$(( $(date +%s) - T0 ))s"

#!/bin/bash
# exp2 = opt 形状位移代价（老引擎 opt=512 vs 新引擎 opt=2048，同输入长度对比）
# exp3 = 真实 2048/4096 TTFT（只有新引擎能到）
# expB = decode 200 迭代，收窄 ±20% 的 run-to-run 方差
A=/home/vision/chenrunbin/alpamayo-edge
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
BIN=/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_bench
OUT=$A/bench/len4096_sweep.csv
RAW=$A/logs/len4096_sweep_raw.log
echo "engine,opt_shape,mode,len,iters,e2e_ms,tok_s,peak_mem_MB" > "$OUT"
: > "$RAW"

bench(){  # tag engdir opt mode flag len iters
  tag=$1; eng=$2; opt=$3; mode=$4; flag=$5; len=$6; it=$7
  ( while true; do free -m | awk '/Mem/{print $3}'; sleep 1; done > /tmp/m.log ) & MON=$!
  o=$("$BIN" --engineDir "$eng" --mode "$mode" "$flag" "$len" --iterations "$it" --warmup 3 2>&1)
  kill $MON 2>/dev/null
  pk=$(sort -n /tmp/m.log | tail -1)
  { echo "########## $tag $mode len=$len iters=$it peak=${pk}MB"; echo "$o" | grep logResultsSummary; } >> "$RAW"
  e=$(echo "$o" | grep -oE "E2E Time \(actual performance\): [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  t=$(echo "$o" | grep -oE "Tokens/sec \(E2E\): [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "$e" ] && { e=FAIL; echo "$o" | tail -20 >> "$RAW"; }
  echo "$tag,$opt,$mode,$len,$it,$e,$t,$pk" | tee -a "$OUT"
}

echo "=== exp2: opt 位移代价 $(date +%T) ==="
for L in 128 512 1024; do
  bench int4_old     $A/engines/int4/llm          512 prefill --inputLen $L 10
  bench int4_len4096 $A/engines/int4_len4096/llm 2048 prefill --inputLen $L 10
  bench int8_old     $A/engines/int8/llm          512 prefill --inputLen $L 10
  bench int8_len4096 $A/engines/int8_len4096/llm 2048 prefill --inputLen $L 10
done
echo "=== exp3: 2048 / 4096 TTFT $(date +%T) ==="
for L in 2048 4096; do
  bench int4_len4096 $A/engines/int4_len4096/llm 2048 prefill --inputLen $L 10
  bench int8_len4096 $A/engines/int8_len4096/llm 2048 prefill --inputLen $L 10
done
echo "=== expB: decode 200 迭代 $(date +%T) ==="
bench int4_old     $A/engines/int4/llm          512 decode --pastKVLen 512 200
bench int8_old     $A/engines/int8/llm          512 decode --pastKVLen 512 200
bench int4_len4096 $A/engines/int4_len4096/llm 2048 decode --pastKVLen 512 200
bench int8_len4096 $A/engines/int8_len4096/llm 2048 decode --pastKVLen 512 200
echo "SWEEP_DONE $(date +%T)"

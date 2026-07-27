#!/bin/bash
# 2026-07-27 路径迁移: /home/nvidia/chenrunbin.crb -> /home/nvidia/chenrunbin/alpamayo-edge
#   build_orin -> trtedge/ , e_int* -> onnx/ , *.tgz -> tarballs/ , build 日志 -> logs/
cd /home/nvidia/chenrunbin/alpamayo-edge || exit 1
export EDGELLM_PLUGIN_PATH=/home/nvidia/chenrunbin/alpamayo-edge/trtedge/build_orin/libNvInfer_edgellm_plugin.so
BENCH=trtedge/build_orin/examples/llm/llm_bench
run_power(){
  tag=$1; shift
  tegrastats --interval 200 > /tmp/tegra_$tag.log 2>&1 &
  TP=$!
  $BENCH "$@" > /tmp/bench_$tag.log 2>&1
  sleep 1; kill $TP 2>/dev/null
  # mean of (GPU_SOC + CPU_CV + SYS_5V0) mW over samples with GPU_SOC>3000 (active)
  awk '{
    for(i=1;i<=NF;i++){
      if($i=="VDD_GPU_SOC"){split($(i+1),a,"/"); g=a[1]+0}
      if($i=="VDD_CPU_CV"){split($(i+1),a,"/"); c=a[1]+0}
      if($i=="VIN_SYS_5V0"){split($(i+1),a,"/"); s=a[1]+0}
    }
    if(g>3000){tot=g+c+s; sum+=tot; n++; if(tot>mx)mx=tot}
  } END{ if(n>0) printf "active_samples=%d mean_W=%.1f peak_W=%.1f\n", n, sum/n/1000, mx/1000; else print "no active samples" }' /tmp/tegra_$tag.log
}
echo "=== FP16 PREFILL power (inputLen512, 200 iters) ==="
run_power fp16pre --engineDir engines/fp16/llm --mode prefill --inputLen 512 --iterations 200 --warmup 5
echo "=== FP16 DECODE power (pastKV512, osl64, 100 iters) ==="
run_power fp16dec --engineDir engines/fp16/llm --mode decode --pastKVLen 512 --osl 64 --iterations 100 --warmup 5

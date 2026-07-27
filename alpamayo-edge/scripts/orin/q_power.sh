#!/bin/bash
# 2026-07-27 路径迁移: /home/nvidia/chenrunbin.crb -> /home/nvidia/chenrunbin/alpamayo-edge
#   build_orin -> trtedge/ , e_int* -> onnx/ , *.tgz -> tarballs/ , build 日志 -> logs/
cd /home/nvidia/chenrunbin/alpamayo-edge || exit 1
export EDGELLM_PLUGIN_PATH=/home/nvidia/chenrunbin/alpamayo-edge/trtedge/build_orin/libNvInfer_edgellm_plugin.so
BENCH=trtedge/build_orin/examples/llm/llm_bench
rp(){ tag=$1; shift; tegrastats --interval 200 > /tmp/tg_$tag.log 2>&1 & TP=$!; $BENCH "$@" >/dev/null 2>&1; sleep 1; kill $TP 2>/dev/null;
  awk '{for(i=1;i<=NF;i++){if($i=="VDD_GPU_SOC"){split($(i+1),a,"/");g=a[1]+0}if($i=="VDD_CPU_CV"){split($(i+1),a,"/");c=a[1]+0}if($i=="VIN_SYS_5V0"){split($(i+1),a,"/");s=a[1]+0}}if(g>3000){t=g+c+s;sum+=t;n++}}END{if(n)printf "mean_W=%.1f\n",sum/n/1000; else print "na"}' /tmp/tg_$tag.log; }
for q in int4 int8; do
  echo "== $q prefill =="; rp ${q}p --engineDir engines/$q/llm --mode prefill --inputLen 512 --iterations 200 --warmup 5
  echo "== $q decode ==";  rp ${q}d --engineDir engines/$q/llm --mode decode --pastKVLen 512 --osl 64 --iterations 100 --warmup 5
done
echo DONE_QPOWER

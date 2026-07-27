#!/bin/bash
# args: <tag> <engineDir> <mode> <lenflag> <lenval> <iters> <warmup>
TAG=$1; ENG=$2; MODE=$3; LF=$4; LV=$5; IT=$6; WU=$7
cd /home/vision/TensorRT-Edge-LLM
export EDGELLM_PLUGIN_PATH=$PWD/build_orin/libNvInfer_edgellm_plugin.so
LOG=/home/vision/chenrunbin/alpamayo-edge/logs/pw_${TAG}.log   # 2026-07-27: 原为 /home/vision/pw_*.log（会污染共享账号的家目录顶层）
tegrastats --interval 250 > $LOG 2>&1 & TS=$!
sleep 1
TPS=$(./build_orin/examples/llm/llm_bench --engineDir $ENG --mode $MODE $LF $LV --iterations $IT --warmup $WU 2>&1 | grep -iE "Tokens/sec" | grep -oE "[0-9.]+$" | tail -1)
kill $TS 2>/dev/null
awk -v tag="$TAG" -v tps="$TPS" "
/GR3D_FREQ/ { gpu=-1;gsoc=0;cpu=0;sys=0;
  for(i=1;i<=NF;i++){
    if(\$i==\"GR3D_FREQ\"){split(\$(i+1),g,\"%\");gpu=g[1]+0}
    if(\$i==\"VDD_GPU_SOC\"){split(\$(i+1),a,\"/\");x=a[1];gsub(/mW/,\"\",x);gsoc=x+0}
    if(\$i==\"VDD_CPU_CV\"){split(\$(i+1),a,\"/\");x=a[1];gsub(/mW/,\"\",x);cpu=x+0}
    if(\$i==\"VIN_SYS_5V0\"){split(\$(i+1),a,\"/\");x=a[1];gsub(/mW/,\"\",x);sys=x+0}
  }
  if(gpu>0){n++;tot=gsoc+cpu+sys;sgs+=gsoc;st+=tot;if(tot>pt)pt=tot}
}
END{if(n>0){atot=st/n/1000; printf \"%s: tok/s=%s | GPU_SOC=%.1fW TOTAL=%.1fW(peak %.1fW) | eff=%.2f tok/J\n\", tag, tps, sgs/n/1000, atot, pt/1000, tps/atot} else print tag\": no active samples\"}
" $LOG

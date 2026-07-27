#!/bin/bash
# 消掉截断 confound：12-prompt 探针与自由问答实例都用 512 的生成上限重跑
# 只改生成长度这一个变量 —— 仍用老引擎 engines/{int4,int8}，与已发布 perplexity 口径一致
A=/home/vision/chenrunbin/alpamayo-edge
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
BIN=/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_inference
CAP=512
python3 - <<PY
import json
for src,dst in (("$A/io/acc_greedy.json","$A/io/acc_greedy_cap512.json"),
                ("$A/io/freeform_in.json","$A/io/freeform_in_cap512.json")):
    d=json.load(open(src)); d["max_generate_length"]=$CAP
    json.dump(d,open(dst,"w"),indent=1)
    print("wrote",dst,"cap=",d["max_generate_length"])
PY
for e in int4 int8; do
  echo "=== probe $e cap=$CAP $(date +%T) ==="
  $BIN --engineDir $A/engines/$e/llm --multimodalEngineDir $A/engines/int4/visual \
       --inputFile $A/io/acc_greedy_cap512.json --outputFile $A/io/acc_${e}_cap512.json \
       --maxGenerateLength $CAP > $A/logs/probe_${e}_cap512.log 2>&1
  echo "  exit=$? -> $(python3 -c "
import json,collections
d=json.load(open('$A/io/acc_${e}_cap512.json'))
r=d['responses']
print(collections.Counter(x.get('finish_reason') for x in r), 'maxchars', max(len(x.get('output_text') or '') for x in r))")"
done
echo "=== freeform cap=$CAP $(date +%T) ==="
$BIN --engineDir $A/engines/int4/llm --multimodalEngineDir $A/engines/int4/visual \
     --inputFile $A/io/freeform_in_cap512.json --outputFile $A/io/freeform_out_cap512.json \
     --maxGenerateLength $CAP > $A/logs/freeform_cap512.log 2>&1
echo "  exit=$? -> $(python3 -c "
import json
d=json.load(open('$A/io/freeform_out_cap512.json'))
x=d['responses'][0]; print(x.get('finish_reason'), len(x.get('output_text') or ''),'chars')")"
echo RERUN_DONE $(date +%T)

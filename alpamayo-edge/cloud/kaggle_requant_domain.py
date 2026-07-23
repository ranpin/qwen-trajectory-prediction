# Kaggle — re-quantize Cosmos-Reason2-8B INT8(SmoothQuant) with an IN-DOMAIN
# calibration set (embodied-reasoning questions from Cosmos-Reason1-Benchmark
# subsets NOT used in eval: agibot/bridgev2/holoassist/robofail), to test whether
# the default news-domain calibration (cnn_dailymail) was suboptimal (ablation).
# No overlap with the robovqa eval subset -> no leakage.
import os, subprocess, glob, shutil, json
assert os.environ.get("HF_TOKEN"); os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
def sh(c): print("+",c,flush=True); subprocess.run(c,shell=True,check=True)
def df(t): print(f"--- df[{t}] ---",flush=True); subprocess.run("df -h /kaggle/working /tmp 2>/dev/null|sort -u",shell=True)

df("start")
sh("git clone --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git /kaggle/working/TRTEdge")
sh("cd /kaggle/working/TRTEdge && pip -q install '.[tools]'")

# --- build in-domain calibration json from benchmark subset questions (small JSONs only, no videos) ---
from huggingface_hub import snapshot_download
CAL_SUBSETS=["agibot","bridgev2","holoassist","robofail"]  # NOT robovqa (eval set)
root=snapshot_download("nvidia/Cosmos-Reason1-Benchmark", repo_type="dataset",
      allow_patterns=[f"{s}/*qa_pairs*.json" for s in CAL_SUBSETS])
texts=[]
for s in CAL_SUBSETS:
    for jf in glob.glob(os.path.join(root,s,"*qa_pairs*.json")):
        data=json.load(open(jf)); recs=data if isinstance(data,list) else list(data.values())
        for r in recs:
            qp=r.get("qa_pairs");
            for qa in (qp if isinstance(qp,list) else [qp]):
                if qa and qa.get("question"): texts.append(qa["question"].strip())
print(f"[calib] collected {len(texts)} in-domain question texts",flush=True)
# concatenate ~4 questions per sample for more representative sequence length
CAL="/kaggle/working/domain_calib.json"
with open(CAL,"w") as f:
    for i in range(0,len(texts),4):
        f.write(json.dumps({"text":"  ".join(texts[i:i+4])})+"\n")
nseq=sum(1 for _ in open(CAL)); print(f"[calib] wrote {nseq} calibration sequences -> {CAL}",flush=True)

# --- patch quantize.py for multi-GPU load + cpu-consolidate export (same as int4/int8 runs) ---
QP="/kaggle/working/TRTEdge/tensorrt_edgellm/quantization/quantize.py"; _src=open(QP).read()
A0=('                model = factory.from_pretrained(\n                    model_dir,\n                    torch_dtype=torch_dtype,\n                    trust_remote_code=True,\n                ).to(device)')
A1=('                model = factory.from_pretrained(\n                    model_dir,\n                    torch_dtype=torch_dtype,\n                    trust_remote_code=True,\n                    device_map="auto",\n                )')
B0='    model.to(torch_dtype)'
B1=('    try:\n        model.to(torch_dtype)\n    except Exception as _e:\n        print("[patch] skip .to(dtype):",_e,flush=True)')
C0=('    os.makedirs(output_dir, exist_ok=True)\n    with torch.inference_mode(), _skip_resmooth_for_hybrid(')
C1=('    _dm=getattr(model,"hf_device_map",None)\n    if _dm and len({str(v) for v in _dm.values()})>1:\n        try:\n            from accelerate.hooks import remove_hook_from_module\n            remove_hook_from_module(model,recurse=True)\n        except Exception as _e: print("[patch] remove_hook:",_e,flush=True)\n        model.to("cpu")\n        try: model.hf_device_map={"":"cpu"}\n        except Exception: pass\n        import gc as _g; _g.collect(); torch.cuda.empty_cache()\n        print("[patch] consolidated sharded->CPU",flush=True)\n\n    os.makedirs(output_dir, exist_ok=True)\n    with torch.inference_mode(), _skip_resmooth_for_hybrid(')
assert _src.count(A0)==1 and _src.count(B0)==1 and _src.count(C0)==1, "patch needle changed"
open(QP,"w").write(_src.replace(A0,A1).replace(B0,B1).replace(C0,C1))
subprocess.run(f"python -m py_compile {QP}",shell=True,check=True); print("[patch] applied",flush=True)

os.environ["PYTHONPATH"]="/kaggle/working/TRTEdge"; os.chdir("/kaggle/working")
MODEL="nvidia/Cosmos-Reason2-8B"; QF="int8_sq"; SCRATCH="/tmp/edge"; ART=f"{SCRATCH}/artifacts"; os.makedirs(ART,exist_ok=True)
ok=False
try:
    ckpt=f"{SCRATCH}/Cosmos-8B-{QF}-domain"; onnx=f"{ckpt}/onnx"
    sh(f"tensorrt-edgellm-quantize llm --model_dir {MODEL} --output_dir {ckpt} --quantization {QF} --dataset {CAL} --num_samples 512")
    df("after-quant"); sh(f"tensorrt-edgellm-export {ckpt} {onnx}"); df("after-onnx")
    dest=f"{ART}/Cosmos-8B-{QF}-domain-onnx-llm"; shutil.move(f"{onnx}/llm",dest); shutil.rmtree(ckpt,ignore_errors=True); df("after-shrink")
    print("DOMAIN_INT8_OK ->",dest,flush=True); ok=True
except Exception as e: print("DOMAIN_INT8_FAILED:",e,flush=True)
finally:
    prod=glob.glob(f"{ART}/*-onnx-llm")
    if prod:
        shutil.rmtree("/kaggle/working/TRTEdge",ignore_errors=True)
        sh("tar -czf /kaggle/working/edge_int8_domain_llm.tgz -C "+ART+" "+os.path.basename(prod[0])); print("PACKED_OK",flush=True)
    else: print("NO_ONNX",flush=True)
    df("end")
print("DONE ok="+str(ok),flush=True)

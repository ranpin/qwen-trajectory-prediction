# Kaggle script kernel — Cosmos-Reason2-8B FP16 (no-quant) ONNX export for the
# Orin performance baseline. Same scaffolding as the int4/int8 export, but with
# NO --quantization (the CLI default = full fp16 backbone). We keep ONLY the LLM
# ONNX (the ~16GB part) to minimize transfer; the fp16 visual tower already exists
# on Orin (shared, unchanged) and isn't needed for the LLM perf comparison.
#
# PREREQ: GPU T4x2 (push with machine_shape=NvidiaTeslaT4), Internet on, HF_TOKEN.
import os, subprocess, glob, shutil
assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set"
os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def sh(c):
    print("+", c, flush=True); subprocess.run(c, shell=True, check=True)


def df(tag):
    print(f"--- df [{tag}] ---", flush=True)
    subprocess.run("df -h /kaggle/working /tmp 2>/dev/null | sort -u", shell=True)


df("start")
sh("git clone --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git /kaggle/working/TRTEdge")
sh("cd /kaggle/working/TRTEdge && pip -q install '.[tools]'")

# Patch quantize.py: (A) multi-GPU load, (B) tolerate .to(dtype), (C) CPU-consolidate
# a sharded model before export. 8B fp16 (16GB) is spread across the two T4s by
# device_map="auto", then consolidated to CPU for a clean ONNX export.
QP = "/kaggle/working/TRTEdge/tensorrt_edgellm/quantization/quantize.py"
_src = open(QP).read()
_A_old = ('                model = factory.from_pretrained(\n'
          '                    model_dir,\n'
          '                    torch_dtype=torch_dtype,\n'
          '                    trust_remote_code=True,\n'
          '                ).to(device)')
_A_new = ('                model = factory.from_pretrained(\n'
          '                    model_dir,\n'
          '                    torch_dtype=torch_dtype,\n'
          '                    trust_remote_code=True,\n'
          '                    device_map="auto",\n'
          '                )')
_B_old = '    model.to(torch_dtype)'
_B_new = ('    try:\n'
          '        model.to(torch_dtype)\n'
          '    except Exception as _edge_e:\n'
          '        print("[edge-patch] skip model.to(dtype) under device_map:", _edge_e, flush=True)')
_C_old = ('    os.makedirs(output_dir, exist_ok=True)\n'
          '    with torch.inference_mode(), _skip_resmooth_for_hybrid(')
_C_new = ('    _dm = getattr(model, "hf_device_map", None)\n'
          '    if _dm and len({str(v) for v in _dm.values()}) > 1:\n'
          '        try:\n'
          '            from accelerate.hooks import remove_hook_from_module\n'
          '            remove_hook_from_module(model, recurse=True)\n'
          '        except Exception as _edge_e:\n'
          '            print("[edge-patch] remove_hook failed:", _edge_e, flush=True)\n'
          '        model.to("cpu")\n'
          '        try:\n'
          '            model.hf_device_map = {"": "cpu"}\n'
          '        except Exception:\n'
          '            pass\n'
          '        import gc as _gc\n'
          '        _gc.collect(); torch.cuda.empty_cache()\n'
          '        print("[edge-patch] consolidated sharded model -> CPU for export", flush=True)\n'
          '\n'
          '    os.makedirs(output_dir, exist_ok=True)\n'
          '    with torch.inference_mode(), _skip_resmooth_for_hybrid(')
assert _src.count(_A_old) == 1 and _src.count(_B_old) == 1 and _src.count(_C_old) == 1, "patch needle changed upstream"
open(QP, "w").write(_src.replace(_A_old, _A_new).replace(_B_old, _B_new).replace(_C_old, _C_new))
subprocess.run(f"python -m py_compile {QP}", shell=True, check=True)
print("[edge-patch] A+B+C applied OK", flush=True)

os.environ["PYTHONPATH"] = "/kaggle/working/TRTEdge"
os.chdir("/kaggle/working")
subprocess.run("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader", shell=True)

MODEL = "nvidia/Cosmos-Reason2-8B"
SCRATCH = "/tmp/edge"; ART = f"{SCRATCH}/artifacts"
os.makedirs(ART, exist_ok=True)
ok = False
try:
    ckpt = f"{SCRATCH}/Cosmos-8B-fp16"; onnx = f"{ckpt}/onnx"
    # NO --quantization  => full fp16 backbone (CLI default)
    sh(f"tensorrt-edgellm-quantize llm --model_dir {MODEL} --output_dir {ckpt}")
    df("after-prep")
    sh(f"tensorrt-edgellm-export {ckpt} {onnx}")
    df("after-onnx")
    # keep ONLY the llm ONNX (the ~16GB part); visual tower already on Orin
    dest = f"{ART}/Cosmos-8B-fp16-onnx-llm"
    shutil.move(f"{onnx}/llm", dest)
    shutil.rmtree(ckpt, ignore_errors=True)
    df("after-shrink")
    print(f"COSMOS_fp16_OK -> {dest}", flush=True); ok = True
except Exception as e:
    print("COSMOS_fp16_FAILED:", e, flush=True)
finally:
    produced = glob.glob(f"{ART}/*-onnx-llm")
    if produced:
        try:
            shutil.rmtree("/kaggle/working/TRTEdge", ignore_errors=True)
            sh("tar -czf /kaggle/working/edge_fp16_llm.tgz -C " + ART + " " +
               " ".join(os.path.basename(p) for p in produced))
            print("PACKED_OK", [os.path.basename(p) for p in produced], flush=True)
        except Exception as pe:
            print("PACK_FAILED:", pe, flush=True)
    else:
        print("NO_ONNX_PRODUCED", flush=True)
    df("end")
shutil.rmtree(ART, ignore_errors=True)
print("DONE ok=" + str(ok), flush=True)

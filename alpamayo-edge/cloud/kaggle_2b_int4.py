# Kaggle script - Cosmos-Reason2-*2B* INT4 (AWQ) export, for the model-size Pareto.
#
# WHY: the 8B deployment answers "which precision", not "which size". A 2B point gives the
# accuracy-vs-latency-vs-power Pareto that a deployment decision actually needs.
# Config deliberately MIRRORS the shipped 8B artifact (int4_awq backbone, lm_head left at
# fp16) so the only variable is model size.
#
# Analytical prediction to check against measurement (2B: 28 layers, hidden 2048, FFN 6144,
# GQA 16/8, vocab 151936):
#   backbone matmul 1.409 B params -> 0.705 GB @int4 ; lm_head 311 M -> 0.622 GB @fp16
#   => engine ~1.355 GB, of which the *unquantized lm_head is 46%* (vs 25.6% on 8B)
#   => decode is DRAM-bound, so TPOT ~ 1.355/4.853 of the 8B's 33.1 ms => ~9-10 ms (~100 tok/s)
#
# Same patches as the 8B run (device_map sharding is harmless for a model this small).
# PREREQ: metadata MUST set machine_shape=NvidiaTeslaT4; Internet = On.
import os, subprocess, glob, shutil

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# HF token resolution order: Kaggle Notebook Secret -> env HF_TOKEN.
# NOTE: never hardcode a token here. For headless API runs where the secret is
# unreadable (the v13 "secret HTTP 400" case), set HF_TOKEN in the environment
# before launching, e.g. via `kaggle kernels push` with the token exported.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("[hf] using Notebook Secret HF_TOKEN", flush=True)
except Exception as _se:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN unavailable: no Kaggle secret and no HF_TOKEN env var. "
            "Set the HF_TOKEN Notebook Secret, or export HF_TOKEN before running."
        ) from _se
    print("[hf] secret unavailable, using env HF_TOKEN:", _se, flush=True)
os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])


def sh(c):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True, check=True)


def df(tag):
    print(f"--- df [{tag}] ---", flush=True)
    subprocess.run("df -h /kaggle/working /root /tmp 2>/dev/null | sort -u", shell=True)


# 1) Fetch + install TRTEdge
df("start")
sh("git clone --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git /kaggle/working/TRTEdge")
sh("cd /kaggle/working/TRTEdge && pip -q install '.[tools]'")

# 2) Patch quantize.py: (A) multi-GPU load, (B) tolerate .to(dtype), (C) CPU-consolidate
#    a sharded model before export. Each needle is asserted unique so an upstream
#    change can't make us believe we patched when we didn't.
QP = "/kaggle/working/TRTEdge/tensorrt_edgellm/quantization/quantize.py"
_src = open(QP).read()
_A_old = (
    '                model = factory.from_pretrained(\n'
    '                    model_dir,\n'
    '                    torch_dtype=torch_dtype,\n'
    '                    trust_remote_code=True,\n'
    '                ).to(device)'
)
_A_new = (
    '                model = factory.from_pretrained(\n'
    '                    model_dir,\n'
    '                    torch_dtype=torch_dtype,\n'
    '                    trust_remote_code=True,\n'
    '                    device_map="auto",\n'
    '                )'
)
_B_old = '    model.to(torch_dtype)'
_B_new = (
    '    try:\n'
    '        model.to(torch_dtype)\n'
    '    except Exception as _edge_e:\n'
    '        print("[edge-patch] skip model.to(dtype) under device_map:", _edge_e, flush=True)'
)
# NOTE (2026-07-26): upstream inserted a comment block between `os.makedirs` and the
# `with torch.inference_mode()` line, so the old two-line needle stopped matching and the
# assert killed a run (working as designed - see docs/PROBLEMS.md G1). Anchor on the single
# `os.makedirs(output_dir, ...)` line instead, which is still unique in the file.
_C_old = '    os.makedirs(output_dir, exist_ok=True)\n'
_C_new = (
    '    _dm = getattr(model, "hf_device_map", None)\n'
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
)
assert _src.count(_A_old) == 1, "PATCH A needle not found — TRTEdge source changed upstream"
assert _src.count(_B_old) == 1, "PATCH B needle not found — TRTEdge source changed upstream"
assert _src.count(_C_old) == 1, "PATCH C needle not found — TRTEdge source changed upstream"
open(QP, "w").write(
    _src.replace(_A_old, _A_new).replace(_B_old, _B_new).replace(_C_old, _C_new)
)
subprocess.run(f"python -m py_compile {QP}", shell=True, check=True)
print("[edge-patch] A(device_map) + B(to-dtype) + C(cpu-export) applied OK", flush=True)

# 3) Environment
os.environ["PYTHONPATH"] = "/kaggle/working/TRTEdge"
os.chdir("/kaggle/working")
subprocess.run("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader", shell=True)

# 4) Quantize -> export ONNX -> immediately shrink footprint. int4_awq only (primary
#    deliverable). int8 is a separate run so it can never clobber int4's disk space.
MODEL = "nvidia/Cosmos-Reason2-2B"
# int4_awq backbone (same as the shipped engine) + lm_head ALSO int4_awq. The only
# delta vs the deployed artifact is the lm_head, so any TPOT/accuracy difference is
# attributable to that single change.
QF = "int4_awq"
LMH = None          # mirror the shipped 8B artifact: lm_head stays fp16
TAG = "2b_int4_awq"
# CRITICAL: /kaggle/working is a fixed 20GB loop device; int8's checkpoint (~10GB)
# + ONNX (~10GB) overflow it. Do all heavy work under /tmp, which lives on the
# 7.9TB overlay `/` (1.1TB free). Only the final packed tgz goes to /kaggle/working
# (the sole persisted kernel output).
SCRATCH = "/tmp/edge"
ART = f"{SCRATCH}/artifacts"
os.makedirs(ART, exist_ok=True)
ok = False
try:
    ckpt = f"{SCRATCH}/Cosmos-{TAG}"
    onnx = f"{ckpt}/onnx"
    sh(f"tensorrt-edgellm-quantize llm --model_dir {MODEL} --output_dir {ckpt} "
       f"--quantization {QF}")
    df("after-quantize")
    sh(f"tensorrt-edgellm-export {ckpt} {onnx}")
    df("after-onnx")
    # keep ONLY the onnx; drop the multi-GB HF checkpoint safetensors at once
    dest = f"{ART}/Cosmos-{TAG}-onnx"
    shutil.move(onnx, dest)
    shutil.rmtree(ckpt, ignore_errors=True)
    df("after-shrink")
    print(f"COSMOS_{TAG}_OK -> {dest}", flush=True)
    ok = True
except Exception as e:
    print(f"COSMOS_{TAG}_FAILED:", e, flush=True)
finally:
    # Always package whatever ONNX we managed to build; never lose a built artifact.
    produced = glob.glob(f"{ART}/*-onnx")
    if produced:
        try:
            # free the HF model cache first so packing has room even if disk is tight
            shutil.rmtree("/kaggle/working/TRTEdge", ignore_errors=True)
            sh("tar -czf /kaggle/working/edge_artifacts_2b.tgz -C " + ART + " " +
               " ".join(os.path.basename(p) for p in produced))
            print("PACKED_OK", [os.path.basename(p) for p in produced], flush=True)
        except Exception as pe:
            print("PACK_FAILED:", pe, flush=True)
    else:
        print("NO_ONNX_PRODUCED", flush=True)
    df("end")

# tidy intermediates (tgz + log remain as the kernel output)
shutil.rmtree(ART, ignore_errors=True)
shutil.rmtree("/kaggle/working/TRTEdge", ignore_errors=True)
print("DONE ok=" + str(ok), flush=True)

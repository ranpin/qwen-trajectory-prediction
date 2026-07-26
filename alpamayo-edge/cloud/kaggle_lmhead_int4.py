# Kaggle script - Cosmos-Reason2-8B INT4 (AWQ) export WITH THE LM-HEAD ALSO QUANTIZED.
#
# WHY THIS RUN EXISTS (measured, not guessed):
#   Nsight profiling of the shipped INT4 engine on Jetson Orin showed the lm_head is
#   NOT quantized -- AWQ skips the output layer by default -- so it sits in the engine
#   as fp16 and accounts for 1.245 GB = 25.6% of every decode step's bytes, and 22.0%
#   of decode time (see alpamayo-edge/eval/profile/ and docs/edge_deploy_status.md).
#   Decode is DRAM-bound (roofline: AI ~3 FLOP/byte vs a 210 ridge), so cutting those
#   bytes should translate almost 1:1 into latency:
#       predicted 4.853 GB -> 3.932 GB (-19%)  =>  TPOT 33.1 ms -> ~26.8 ms (+23% tok/s)
#   TRTEdge exposes this directly: --lm_head_quantization int4_awq (no source patch).
#   Risk to measure, not assume: the output layer is the most quantization-sensitive
#   part of the model, so this run must be re-scored on the n=210 benchmark, not just timed.
#
# History of failures the patches below fix (each was a real run):
#   v13  secret HTTP 400     -> HF token from env/secret (never hardcode in the git copy)
#   v14  CUDA OOM on load    -> device_map="auto" shards 8B across both T4s (PATCH A/B)
#   v15a cross-device export -> consolidate sharded model to CPU before export (PATCH C)
#   v15b "No space left"     -> heavy work under /tmp, drop HF checkpoint right after ONNX
#                               export, pack in try/finally
#
# PREREQ: Accelerator = GPU T4 x2 (metadata MUST set machine_shape=NvidiaTeslaT4; plain
#         enable_gpu gives a single P100/sm_60 which new torch cannot run), Internet = On.
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
# NOTE (2026-07-26): plain device_map="auto" fills both T4s to ~13.6/14.56 GiB, and
# quantizing the lm_head needs a ~1.16 GiB temp for its 4096x151936 weight -> CUDA OOM
# inside modelopt's fake_quantize (run v2, failed at 845 s).
# v3 tried max_memory=11GiB/GPU and STILL OOMed at the same point (836 s) with GPU 1 again
# at 13.58 GiB: the cap bounds only the *initial weight placement*, while quantization then
# adds ~2.5 GiB of its own state (11 + 2.5 ~= 13.5, matching the log) and eats the headroom
# back. So the cap has to be low enough to cover placement AND quantization state:
# v5 (10 GiB/GPU, placement {'0':16,'1':25}, nothing on CPU) OOMed again with the SAME free
# memory (1006.81 MiB) as v3 -> `max_memory` bounds only the initial PLACEMENT; quantization
# state then grows to the same water line regardless of the cap. Capping the total is the
# wrong lever. What matters is the free room on the ONE card that receives lm_head.
# Modules are assigned in order, so lm_head (last) lands on the LAST device -> make that
# device hold few weights: ASYMMETRIC caps. Empirically a card holding W GiB of weights ends
# up at W + ~2.5 GiB, so 11 GiB on GPU0 (-> ~13.5, survived in v3) and 6 GiB on GPU1
# (-> ~8.5, leaving ~6 GiB for the 1.16 GiB lm_head temp). 11+6=17 GiB >= the 16.3 GiB model,
# so nothing spills to CPU (v4 showed CPU offload breaks AWQ calibration outright).
# v4 (9 GiB/GPU) for the record: the cap DID take effect (placement {'0':16,'1':24,'cpu':1}) but
# 18 GiB was not enough for the ~16.3 GiB model + accelerate's accounting, so ONE module
# (the 1.24 GB lm_head, the largest single module) spilled to CPU -- and then AWQ calibration
# died with `CUDA error: an illegal memory access` inside F.linear under accelerate's offload
# hook: ModelOpt's AWQ calibration and CPU offload do not mix.
# So the window is narrow: everything must stay on GPU *and* ~1.2 GiB must stay free on the
# card holding lm_head. 11 GiB -> OOM by 0.16 GiB; 9 GiB -> CPU spill. Use 10 GiB:
# 20 GiB total covers the 16.3 GiB model (no spill) and leaves ~4 GiB/GPU for temporaries.
# The device-map print stays so the placement is verified, never assumed.
_A_new = (
    '                model = factory.from_pretrained(\n'
    '                    model_dir,\n'
    '                    torch_dtype=torch_dtype,\n'
    '                    trust_remote_code=True,\n'
    '                    device_map="auto",\n'
    '                    max_memory={0: "11GiB", 1: "6GiB", "cpu": "60GiB"},\n'
    '                )\n'
    '                try:\n'
    '                    from collections import Counter as _C\n'
    '                    _dm = getattr(model, "hf_device_map", None) or {}\n'
    '                    print("[edge-patch] device placement:", dict(_C(str(v) for v in _dm.values())), flush=True)\n'
    '                except Exception as _e:\n'
    '                    print("[edge-patch] device map probe failed:", _e, flush=True)'
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
MODEL = "nvidia/Cosmos-Reason2-8B"
# int4_awq backbone (same as the shipped engine) + lm_head ALSO int4_awq. The only
# delta vs the deployed artifact is the lm_head, so any TPOT/accuracy difference is
# attributable to that single change.
QF = "int4_awq"
LMH = "int4_awq"
TAG = "int4_awq_lmh"
# CRITICAL: /kaggle/working is a fixed 20GB loop device; int8's checkpoint (~10GB)
# + ONNX (~10GB) overflow it. Do all heavy work under /tmp, which lives on the
# 7.9TB overlay `/` (1.1TB free). Only the final packed tgz goes to /kaggle/working
# (the sole persisted kernel output).
SCRATCH = "/tmp/edge"
ART = f"{SCRATCH}/artifacts"
os.makedirs(ART, exist_ok=True)
ok = False
try:
    ckpt = f"{SCRATCH}/Cosmos-8B-{TAG}"
    onnx = f"{ckpt}/onnx"
    sh(f"tensorrt-edgellm-quantize llm --model_dir {MODEL} --output_dir {ckpt} "
       f"--quantization {QF} --lm_head_quantization {LMH}")
    df("after-quantize")
    sh(f"tensorrt-edgellm-export {ckpt} {onnx}")
    df("after-onnx")
    # keep ONLY the onnx; drop the multi-GB HF checkpoint safetensors at once
    dest = f"{ART}/Cosmos-8B-{TAG}-onnx"
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
            sh("tar -czf /kaggle/working/edge_artifacts_lmhead.tgz -C " + ART + " " +
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

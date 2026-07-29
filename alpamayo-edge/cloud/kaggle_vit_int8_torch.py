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


# ---- 版本指纹（必须最先打印）------------------------------------------
# v9(成功) 与 v5(失败) 的 transformers 版本与 TRTEdge commit **都没被记录**，
# 导致无法对照 —— 从本 run 起一律先打印，别再让"昨天能跑"变成不可复现。
def _fingerprint():
    import importlib
    for _m in ("torch", "transformers", "modelopt", "accelerate", "onnx"):
        try:
            print(f"[ver] {_m} = {importlib.import_module(_m).__version__}", flush=True)
        except Exception as _e:
            print(f"[ver] {_m} = <{type(_e).__name__}>", flush=True)


def sh(c):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True, check=True)


def df(tag):
    print(f"--- df [{tag}] ---", flush=True)
    subprocess.run("df -h /kaggle/working /root /tmp 2>/dev/null | sort -u", shell=True)


# 1) Fetch + install TRTEdge
df("start")
# 2026-07-28: pin 到 **Orin 上正在用的同一个 commit**（METHODOLOGY §0: v0.9.0 @1ac0f2b）。
# 此前一直是 `clone --depth 1` 未 pin —— G1 已因上游漂移吃过一次，且 v5 的失败
# 无法与 v9 对照正是因为两次的版本都没被记录（§4.1 那条可复现性缺口在报账）。
TRTEDGE_PIN = "1ac0f2b"
sh("git clone -q https://github.com/NVIDIA/TensorRT-Edge-LLM.git /kaggle/working/TRTEdge")
sh(f"git -C /kaggle/working/TRTEdge checkout -q {TRTEDGE_PIN}")
sh("git -C /kaggle/working/TRTEdge rev-parse --short HEAD")
_fingerprint()
sh("cd /kaggle/working/TRTEdge && pip -q install '.[tools]'")
# 2026-07-29：**把 transformers 压回 4.x**。v6 的版本指纹拿到了决定性证据：
#   [ver] transformers = 5.0.0，且 ModelOpt 自己 warn "transformers>=5.0 support is experimental"，
#   同时 transformers 5.0 把 `torch_dtype` 弃用（日志里明确提示改用 `dtype`）——而 TRTEdge 传的
#   正是 torch_dtype，这很可能就是 v3/v5 在 modeling_utils.py:3701 的 .to() 上 OOM 的直接来源。
# lm_head v9（两天前成功）几乎肯定跑在 4.x。脚本一直用 `>=4.51` 未设上界 ⇒ 主版本一发布就被带走。
# 用 <5 而不是钉死某个小版本：既排除主版本破坏性变更，又不需要猜 v9 当天的确切版本
# （那个版本没被记录 —— 这正是本次要修的缺口）。实际解析到的版本由 _fingerprint() 记录。
sh("pip -q install 'transformers>=4.57,<5' 2>&1 | tail -2", check=False)
_fingerprint()   # 降级后再打一次指纹，确认真的生效


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
    '                import os as _eos, json as _ejson\n'
    '                _ekw = _ejson.loads(_eos.environ.get("EDGE_LOAD_KW", "{}"))\n'
    '                if "max_memory" in _ekw:\n'
    '                    _ekw["max_memory"] = {(int(k) if str(k).isdigit() else k): v\n'
    '                                          for k, v in _ekw["max_memory"].items()}\n'
    '                print("[edge-patch] load kwargs:", _ekw, flush=True)\n'
    '                model = factory.from_pretrained(\n'
    '                    model_dir,\n'
    '                    torch_dtype=torch_dtype,\n'
    '                    trust_remote_code=True,\n'
    '                    **_ekw,\n'
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
# --- PATCH D：AWQ 校准 batch 16 -> 1 ---------------------------------------
# v3 去掉 D 的结果是在加载/dtype 阶段 CUDA OOM（GPU0 只剩 4.81 MiB）。与 lm_head v9
# （同为 max_memory 12/12、成功）唯一的差别就是这个 batch，故加回来。
# 顺带好处：与 lm_head INT4 那版的量化配方一致，便于对比。
_D_old = '            batch_size = 16 if quantization in (None, "int4_awq") else 1'
_D_new = '            batch_size = 1  # edge-patch D'
assert _src.count(_D_old) == 1, "PATCH D needle not found — TRTEdge source changed upstream"
assert _src.count(_A_old) == 1, "PATCH A needle not found — TRTEdge source changed upstream"
assert _src.count(_B_old) == 1, "PATCH B needle not found — TRTEdge source changed upstream"
assert _src.count(_C_old) == 1, "PATCH C needle not found — TRTEdge source changed upstream"
open(QP, "w").write(
    _src.replace(_A_old, _A_new).replace(_B_old, _B_new).replace(_C_old, _C_new)
       .replace(_D_old, _D_new))
subprocess.run(f"python -m py_compile {QP}", shell=True, check=True)
print("[edge-patch] A(device_map) + B(to-dtype) + C(cpu-export) + D(calib batch=1) applied OK", flush=True)

# ---------------------------------------------------------------------------
# PATCH E：在 quantize 流程里追加"量化 model.visual"（PyTorch 侧，绕开 ORT 跑不了插件的问题）
# 注入点 = PATCH C 的锚点之前，此时 LLM 已量化完、模型仍在内存里、还没保存 checkpoint。
# ---------------------------------------------------------------------------
# 注意：PATCH C 是**插入**这一行的，所以 _E_old 只在 A/B/C 落盘之后才存在。
# 而上面 open(QP,"w").write(...) 只写了文件、`_src` 变量仍是**原始**未打补丁的源码，
# 因此这里必须**重新读回已打补丁的文件**，否则 E 的 needle 必然找不到（v1 就是这么失败的）。
_src = open(QP).read()
# needle 必须唯一：PATCH A 插入的行里也含"4 空格 + _dm = getattr(...)"这一段（它缩进 20 空格，
# 但子串匹配照样命中）=> 只用那一行会 count==2，v1/v2 就是这么被断言拦下的。
# 改为带上 C 特有的下一行，唯一性由 `if _dm and len(...)` 保证。
_E_old = ('    _dm = getattr(model, "hf_device_map", None)\n'
          '    if _dm and len({str(v) for v in _dm.values()}) > 1:')
_E_new = (
    '    # ---- edge-patch E: quantize the vision tower in PyTorch (INT8) ----\n'
    '    try:\n'
    '        import os as _eos2\n'
    '        import modelopt.torch.quantization as _mtq\n'
    '        _vis = getattr(model, "visual", None)\n'
    '        if _vis is None:\n'
    '            _inner = getattr(model, "model", None)\n'
    '            _vis = getattr(_inner, "visual", None) if _inner is not None else None\n'
    '        print("[E] visual module:", type(_vis).__name__ if _vis is not None else None, flush=True)\n'
    '        if _vis is not None:\n'
    '            _nlin = sum(1 for _m in _vis.modules()\n'
    '                        if _m.__class__.__name__ in ("Linear", "Conv3d", "Conv2d"))\n'
    '            print("[E] visual 可量化线性/卷积层数 =", _nlin, flush=True)\n'
    '            _calib_dir = _eos2.environ.get("EDGE_VIT_CALIB", "random")\n'
    '            def _vit_loop(_mod):\n'
    '                import torch as _t\n'
    '                _p = next(_mod.parameters())\n'
    '                _dev, _dt = _p.device, _p.dtype\n'
    '                _batches = []\n'
    '                if _calib_dir != "random":\n'
    '                    try:\n'
    '                        import glob as _g\n'
    '                        from PIL import Image as _I\n'
    '                        from transformers import AutoProcessor as _AP\n'
    '                        _pr = _AP.from_pretrained(model_dir, trust_remote_code=True)\n'
    '                        _fs = sorted(_g.glob(_calib_dir + "/*.jpg"))[:64]\n'
    '                        print("[E] 真实校准帧", len(_fs), "张", flush=True)\n'
    '                        for _f in _fs:\n'
    '                            _o = _pr.image_processor(images=[_I.open(_f).convert("RGB")],\n'
    '                                                    return_tensors="pt")\n'
    '                            _batches.append((_o["pixel_values"].to(_dev, _dt),\n'
    '                                             _o["image_grid_thw"].to(_dev)))\n'
    '                    except Exception as _ce:\n'
    '                        print("[E] 真实帧校准失败，退化随机:", type(_ce).__name__, _ce, flush=True)\n'
    '                        _batches = []\n'
    '                if not _batches:\n'
    '                    print("[E] !! RANDOM 校准 —— 产出只可回答结构问题，不可用于精度", flush=True)\n'
    '                    for _ in range(8):\n'
    '                        _batches.append((_t.randn(256, 1536, device=_dev, dtype=_dt),\n'
    '                                         _t.tensor([[1, 16, 16]], device=_dev)))\n'
    '                _ok = 0\n'
    '                _last = None\n'
    '                for _pv, _gt in _batches:\n'
    '                    for _sig in ("kw", "pos", "single"):\n'
    '                        try:\n'
    '                            if _sig == "kw":\n'
    '                                _mod(_pv, grid_thw=_gt)\n'
    '                            elif _sig == "pos":\n'
    '                                _mod(_pv, _gt)\n'
    '                            else:\n'
    '                                _mod(_pv)\n'
    '                            _ok += 1\n'
    '                            break\n'
    '                        except Exception as _fe:\n'
    '                            _last = _fe\n'
    '                    else:\n'
    '                        print("[E] forward 三种签名全失败:", type(_last).__name__, _last, flush=True)\n'
    '                        break\n'
    '                print("[E] 校准 forward 成功", _ok, "/", len(_batches), flush=True)\n'
    '            _mtq.quantize(_vis, _mtq.INT8_DEFAULT_CFG, forward_loop=_vit_loop)\n'
    '            _nq = sum(1 for _m in _vis.modules() if "Quant" in _m.__class__.__name__)\n'
    '            print("[E] 量化后 visual 里 Quant* 模块数 =", _nq, flush=True)\n'
    '            print("EDGE_VIT_QUANT_OK", flush=True)\n'
    '    except Exception as _ee:\n'
    '        import traceback as _tb\n'
    '        print("EDGE_VIT_QUANT_FAILED:", type(_ee).__name__, _ee, flush=True)\n'
    '        _tb.print_exc()\n'
    '    # -------------------------------------------------------------------\n'
    '    _dm = getattr(model, "hf_device_map", None)\n'
    '    if _dm and len({str(v) for v in _dm.values()}) > 1:'
)
assert _src.count(_E_old) == 1, "PATCH E needle not found — TRTEdge source changed upstream"
open(QP, "w").write(_src.replace(_E_old, _E_new))
print("[edge-patch] E(visual INT8) applied OK", flush=True)

# ---------------------------------------------------------------------------
MODEL = "nvidia/Cosmos-Reason2-8B"
SCRATCH = "/tmp/edge"
ART = f"{SCRATCH}/artifacts"
os.makedirs(ART, exist_ok=True)
# v3/v5 都在 GPU0 上 OOM（剩 4.81 MiB，两次字节级相同 ⇒ 与校准 batch 无关）。
# 改为非对称并给 GPU0 留头寸：10+13 = 23 GiB > 模型 16.34 GiB ⇒ 不会像 G3 v6 那样外溢。
os.environ["EDGE_LOAD_KW"] = '{"device_map": "auto", "max_memory": {"0": "10GiB", "1": "13GiB"}}'
os.environ.setdefault("EDGE_VIT_CALIB", "random")

ok = False
try:
    ckpt = f"{SCRATCH}/Cosmos-8B-vit_int8"
    onnx_out = f"{ckpt}/onnx"
    sh(f"tensorrt-edgellm-quantize llm --model_dir {MODEL} --output_dir {ckpt} --quantization int4_awq")
    df("after-quantize")
    sh(f"tensorrt-edgellm-export {ckpt} {onnx_out}")
    df("after-onnx")
    dest = f"{ART}/Cosmos-8B-vit_int8-onnx"
    shutil.move(onnx_out, dest)
    shutil.rmtree(ckpt, ignore_errors=True)
    ok = True

    print("\n" + "=" * 72, flush=True)
    print("=== 首要问题：导出的 visual/model.onnx 里还有 QDQ 吗", flush=True)
    print("=" * 72, flush=True)
    import onnx as _onnx
    for _p in glob.glob(f"{dest}/**/visual/model.onnx", recursive=True):
        _m = _onnx.load(_p, load_external_data=False)
        _ops = {}
        for _n in _m.graph.node:
            _ops[_n.op_type] = _ops.get(_n.op_type, 0) + 1
        _qdq = {k: v for k, v in _ops.items() if "Quantize" in k}
        print(f"  {_p}", flush=True)
        print(f"  节点总数 {len(_m.graph.node)}   QDQ = {_qdq or '无'}", flush=True)
        print(f"  ViTAttentionPlugin = {_ops.get('ViTAttentionPlugin', 0)}", flush=True)
        print("  IO: " + ", ".join(
            f"{t.name}={_onnx.TensorProto.DataType.Name(t.type.tensor_type.elem_type)}"
            for t in list(_m.graph.input) + list(_m.graph.output)), flush=True)
        _d = _p + ".data"
        if os.path.exists(_d):
            print(f"  权重 {os.path.getsize(_d)/2**20:.0f} MiB（fp16 基线 1104；减半 ≈ 552）", flush=True)
        print("QDQ_SURVIVED_EXPORT" if _qdq else "QDQ_STRIPPED_BY_EXPORT", flush=True)
except Exception as e:
    print("VIT_INT8_FAILED:", type(e).__name__, e, flush=True)
    import traceback
    traceback.print_exc()
finally:
    produced = sorted(glob.glob(f"{ART}/*-onnx"))
    if produced:
        try:
            shutil.rmtree("/kaggle/working/TRTEdge", ignore_errors=True)
            for p in produced:
                cands = glob.glob(os.path.join(p, "**", "visual"), recursive=True)
                if cands:
                    v = cands[0]
                    sh(f"tar -czf /kaggle/working/vit_int8_visual.tgz -C {os.path.dirname(v)} visual")
                    print("PACKED_OK visual only", flush=True)
                    break
        except Exception as pe:
            print("PACK_FAILED:", pe, flush=True)
    else:
        print("NO_ONNX_PRODUCED", flush=True)
    df("end")
shutil.rmtree(ART, ignore_errors=True)
print("DONE ok=" + str(ok), flush=True)

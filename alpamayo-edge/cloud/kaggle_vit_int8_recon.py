#!/usr/bin/env python3
"""ViT INT8 侦察（第 1 次 run）：只回答"管道通不通"，不做长校准。

背景：TRTEdge 的 `--visual_quantization` 只给 fp8 而 Orin 是 sm_87，我此前据此把
"视觉编码器量化"标成不可行 —— **判据不成立**（见 docs/METHODOLOGY.md §3.6）。设备侧侦察已确认：
  * `createBuilderConfig` 只设 kMONITOR_MEMORY，**连 kFP16 都没设** ⇒ 精度由 ONNX 决定；
  * `qwenViTRunner` 把 IO 分配为 kHALF ⇒ 与"IO 保持 fp16、内部层 INT8"的 TRT 显式量化形态兼容。
剩下的唯一未知在量化工具链这一侧，本脚本就查它。

本 run 刻意**不**做真实校准（那要先解决 ViT 输入张量的构造，见下），只按"最可能致命者优先"跑四问：
  Q1 ModelOpt 的 ONNX 量化 API 在这个环境里有没有、叫什么
  Q2 TRTEdge 导出的 visual/model.onnx 的**输入签名**（名字 / 形状 / dtype）是什么
      —— 这决定真实校准数据要怎么造，是下一次 run 的前提
  Q3 用**随机数据**跑一次 int8 ONNX PTQ 能否产出带 QDQ 的 ONNX（纯管道验证）
  Q4 产出的 ONNX 的 IO dtype 是否仍为 fp16（若被改成 int8，运行时的 kHALF 假设就会崩）

⚠️ 本 run 产出的 ONNX **绝不可用于精度评测**：随机校准的激活范围是垃圾。
   它只用来回答"工具链能不能产出这种 ONNX"。真实校准（用 n=210 基准的 1260 张实帧）
   放到下一次 run，前提是 Q2 给出输入签名。

PREREQ: GPU T4 x2（metadata 必须设 machine_shape=NvidiaTeslaT4），Internet On。
"""
import os
import subprocess
import sys
import traceback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN unavailable: set it via env before push")

MODEL = "nvidia/Cosmos-Reason2-8B"
SCRATCH = "/tmp/vit"
os.makedirs(SCRATCH, exist_ok=True)


def sh(c, check=True):
    print(f"\n$ {c}", flush=True)
    r = subprocess.run(c, shell=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode}: {c}")
    return r.returncode


def section(t):
    print(f"\n{'='*72}\n=== {t}\n{'='*72}", flush=True)


# ---------------------------------------------------------------- Q1
section("Q1  ModelOpt 的 ONNX 量化 API")
sh("pip -q install nvidia-modelopt[onnx] onnxruntime 2>&1 | tail -3", check=False)
api = None
try:
    import modelopt
    print("modelopt version:", getattr(modelopt, "__version__", "?"), flush=True)
    import modelopt.onnx.quantization as moq
    api = [n for n in dir(moq) if not n.startswith("_")]
    print("modelopt.onnx.quantization 导出的名字:", api, flush=True)
    print("Q1_OK", flush=True)
except Exception as e:
    print("Q1_BLOCKED:", type(e).__name__, e, flush=True)
    traceback.print_exc()

# ---------------------------------------------------------------- Q2
section("Q2  TRTEdge 导出的 visual/model.onnx 输入签名")
onnx_dir = None
try:
    sh("git clone -q --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM /kaggle/working/TRTEdge")
    sh("pip -q install -e '/kaggle/working/TRTEdge[tools]' 2>&1 | tail -3", check=False)
    # 只要视觉塔的 ONNX：不量化 LLM，直接 export（视觉本来就是 fp16 导出）
    ck = f"{SCRATCH}/ckpt"
    sh(f"tensorrt-edgellm-quantize llm --model_dir {MODEL} --output_dir {ck} --quantization int4_awq")
    sh(f"tensorrt-edgellm-export {ck} {SCRATCH}/onnx")
    import glob
    cands = glob.glob(f"{SCRATCH}/onnx/**/visual/model.onnx", recursive=True)
    print("找到视觉 ONNX:", cands, flush=True)
    if cands:
        onnx_dir = os.path.dirname(cands[0])
        import onnx
        m = onnx.load(cands[0], load_external_data=False)
        print("\n--- inputs ---", flush=True)
        for i in m.graph.input:
            t = i.type.tensor_type
            dims = [(d.dim_param or d.dim_value) for d in t.shape.dim]
            print(f"  {i.name:28} dtype={onnx.TensorProto.DataType.Name(t.elem_type):8} shape={dims}", flush=True)
        print("--- outputs ---", flush=True)
        for o in m.graph.output:
            t = o.type.tensor_type
            dims = [(d.dim_param or d.dim_value) for d in t.shape.dim]
            print(f"  {o.name:28} dtype={onnx.TensorProto.DataType.Name(t.elem_type):8} shape={dims}", flush=True)
        ops = {}
        for n in m.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        print("\n算子直方图(top12):", sorted(ops.items(), key=lambda x: -x[1])[:12], flush=True)
        print("已有 QDQ?", {k: v for k, v in ops.items() if "Quantize" in k} or "无", flush=True)
        print("Q2_OK", flush=True)
    else:
        print("Q2_BLOCKED: 导出目录里找不到 visual/model.onnx", flush=True)
except Exception as e:
    print("Q2_BLOCKED:", type(e).__name__, e, flush=True)
    traceback.print_exc()

# ---------------------------------------------------------------- Q3 / Q4
section("Q3  用随机数据跑一次 int8 ONNX PTQ（纯管道验证，结果不可用于精度）")
if onnx_dir and api:
    try:
        import numpy as np
        import onnx
        src = os.path.join(onnx_dir, "model.onnx")
        m = onnx.load(src, load_external_data=False)
        calib = {}
        for i in m.graph.input:
            t = i.type.tensor_type
            dims = [d.dim_value if d.dim_value > 0 else 64 for d in t.shape.dim]
            npdt = {1: np.float32, 10: np.float16, 6: np.int32, 7: np.int64}.get(t.elem_type, np.float32)
            arr = (np.random.randn(*dims).astype(npdt) if npdt in (np.float32, np.float16)
                   else np.zeros(dims, dtype=npdt))
            calib[i.name] = arr
            print(f"  随机校准张量 {i.name}: {arr.shape} {arr.dtype}", flush=True)
        out = f"{SCRATCH}/visual_int8.onnx"
        import modelopt.onnx.quantization as moq
        fn = getattr(moq, "quantize", None)
        print("调用:", fn, flush=True)
        fn(onnx_path=src, calibration_data=calib, output_path=out, quantize_mode="int8")
        print("Q3_OK ->", out, os.path.getsize(out), "bytes", flush=True)

        section("Q4  产出 ONNX 的 IO dtype 是否仍为 fp16")
        q = onnx.load(out, load_external_data=False)
        for i in list(q.graph.input) + list(q.graph.output):
            print(f"  {i.name:28} {onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type)}", flush=True)
        ops = {}
        for n in q.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        print("QDQ 节点数:", {k: v for k, v in ops.items() if "Quantize" in k} or "无", flush=True)
        os.makedirs("/kaggle/working/out", exist_ok=True)
        sh(f"cp {out} /kaggle/working/out/ && ls -la /kaggle/working/out", check=False)
        print("Q4_OK", flush=True)
    except Exception as e:
        print("Q3_OR_Q4_BLOCKED:", type(e).__name__, e, flush=True)
        traceback.print_exc()
else:
    print("SKIPPED（Q1 或 Q2 未通过）", flush=True)

print("\nVIT_RECON_DONE", flush=True)

# 边缘量化部署 — 端到端状态（里程碑固化）

> 记录 Cosmos-Reason2-8B 从云端量化到 Jetson Orin 建引擎的完整链路结果，以及当前的运行时已知限制。
> 更新日期：2026-07-20。

## 一句话结论

**量化链路全线跑通**：`nvidia/Cosmos-Reason2-8B` 已产出 **INT4 (AWQ)** 与 **INT8 (SmoothQuant)** 两份边缘 ONNX 产物；INT4 在 Orin 上 **TensorRT 引擎构建 + 加载均成功**。
唯一未通的是**推理运行时**——被 TensorRT-Edge-LLM 预编译插件在 **sm_87 (Orin)** 上的一个 FMHA kernel 覆盖缺口挡住（详见「已知限制」）。此问题与量化无关。

## 已验证跑通的链路

```
Kaggle(T4x2) 量化+导出 ONNX  →  打包 tgz  →  下载/校验  →  scp 到 Orin  →  TRTEdge 建引擎  →  加载
   ✅ INT4/INT8            ✅          ✅ SHA256     ✅            ✅ LLM+视觉塔    ✅
                                                                              推理 ❌（FMHA kernel 缺口）
```

### 关键工程坑与解法（全部已解决）
| 现象 | 根因 | 解法 |
|---|---|---|
| secret HTTP 400 | API 无头运行读不到 Kaggle Notebook Secret | 脚本内嵌 HF token 回退 |
| CUDA OOM（加载） | 8B fp16(~16GB) 硬塞单块 15GB T4 | `device_map="auto"` 跨双 T4 分片加载 |
| 导出 illegal memory access | modelopt 导出假设单设备，模型却分片跨 cuda:0/1 | 导出前摘 accelerate hook、收拢到 CPU |
| `No space left`（导出/打包） | `/kaggle/working` 固定 20GB，装不下 checkpoint+onnx | 重活挪到 `/tmp`（7.9T 大盘），只把 tgz 写回 working |

（量化脚本：[`cloud/kaggle_cosmos_only.py`](../cloud/kaggle_cosmos_only.py)，含上述三处对 TRTEdge `quantize.py` 的补丁。）

## 产物体积对照

| 组件 | INT4 (AWQ) | INT8 (SmoothQuant) | 比值 |
|---|---|---|---|
| LLM ONNX 权重 (`llm/model.onnx.data`) | 4.83 GB | 8.20 GB | 1.70× |
| 词嵌入 (`embedding.safetensors`, fp16) | 1.24 GB | 1.24 GB | 1.00× |
| 视觉塔 ONNX (`visual/model.onnx.data`, fp16) | 1.16 GB | 1.16 GB | 1.00× |
| 打包 tgz | 5.77 GB | 7.81 GB | 1.35× |
| **Orin TensorRT 引擎** — LLM | 4.85 GB | 8.25 GB | 1.70× |
| **Orin TensorRT 引擎** — 视觉塔 | 1.1 GB | 同 INT4（同一份 fp16 视觉塔，不重建） | 1.00× |

> INT8 产物 SHA256 `66f05e3872a5288ba336eff478e5009f9ef850aaa5caa60f4392bb7032bb5db3`（`outputs/edge_artifacts_int8_sq.tgz`，7,807,211,181 B）。
> 主干 LLM 权重 INT8 是 INT4 的 1.70×，与 W8A8 vs W4A16 的理论 2× 接近（差在两版共享的 fp16 词嵌入/lm_head 未随之翻倍）。

> 说明：视觉塔与词嵌入两种量化下相同——工具的视觉塔当前仅支持 fp8，本次两版都保留 fp16；主干 LLM 才是量化差异所在。

## 引擎/模型规格（从 Orin 引擎 config 读出）
- 架构：Qwen2.5 系（Cosmos-Reason 主干），`hiddenSize=4096`、`numDecoderLayers=36`、`numKVHeads=8`（GQA）、`headDim=128`、`vocabSize=151936`。
- 引擎构建参数：`maxInputLen=1024`、`maxKVCapacity=4096`、`maxBatch=4`。
- 加载正常：81 个 I/O 张量、上显存约 4.6 GB。

## 已知限制（BLOCKER，运行时）

**症状**：`llm_bench` / 推理一进 context attention 就抛
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  There must be one kernel to implement the MHA
```

**定位**：`cpp/kernels/contextAttentionKernels/contextFMHARunner.cpp:458`
```cpp
check::check(kernelInfo.mSharedMemBytes != 0, "There must be one kernel to implement the MHA");
```
运行时用 `{dataType, paddedSeqLen, headSize=128, ..., mask=CAUSAL, layout=SEPARATE_Q_K_V, tiling}` 组成的 hashKey 去查预编译 FMHA kernel 表，在 **sm_87** 上取到空 kernel（`mSharedMemBytes==0`）。

**根因**：预编译 FMHA cubin（`cpp/kernels/contextAttentionKernels/cubin/*.cubin.cpp`）在 sm_87 上对该 headSize/mask/tiling **组合有覆盖缺口**。sm_87 存在 `..._S_q_k_v_128_sm87`（分离布局 headSize128），但运行时实际请求的具体组合未命中。与量化、与我们的产物无关，纯属 TRTEdge 该版本插件的 kernel 覆盖问题。

**已在两种量化上复现（坐实与量化无关）**：INT4 与 INT8 引擎**均构建成功**（build 阶段日志还打印 `AttentionPlugin: FMHA supported for headSize=128`），但 `llm_bench --mode prefill` 一进 attention warmup 就在**同一处**崩（`what(): There must be one kernel to implement the MHA`，core dump）。两版症状、堆栈完全一致 → 确为架构/运行时的 kernel 覆盖问题，而非某个量化路径的产物问题。

**未来若要解**（择一，均有不确定性）：
1. 插桩打印运行时确切 hashKey，比对 cubin 表，确认缺失的确切组合；
2. 尝试改 `llm_build` 参数（`maxInputLen`/batch）或真实 `llm_inference` 路径，看能否落到已存在的 kernel；
3. 用 NVIDIA 的 fmha_v2 kernel 生成器补编 sm_87 缺失变体后重编插件；
4. 上游提 issue。

## 复现命令（Orin）
```bash
cd /home/vision/TensorRT-Edge-LLM
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
# 建引擎（成功）
./build/examples/llm/llm_build   --onnxDir /home/vision/edge_int4/Cosmos-8B-int4_awq-onnx/llm    --engineDir /home/vision/engines/int4/llm
./build/examples/multimodal/visual_build --onnxDir /home/vision/edge_int4/Cosmos-8B-int4_awq-onnx/visual --engineDir /home/vision/engines/int4
# 跑推理（触发已知限制）
./build/examples/llm/llm_bench   --engineDir /home/vision/engines/int4/llm --mode prefill --inputLen 512
```

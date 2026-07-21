# 边缘量化部署 — 端到端状态（Orin 推理跑通）

> 记录 Cosmos-Reason2-8B 从云端量化到 Jetson Orin 端到端**推理跑通**的完整链路、功能/性能指标，以及全过程问题与解决方案。
> 更新日期：2026-07-21。

## 一句话结论

**全链路跑通，Orin 上真实出 token**：`nvidia/Cosmos-Reason2-8B` 已产出 **INT4 (AWQ)** 与 **INT8 (SmoothQuant)** 两份边缘产物，两版均在 Jetson Orin (sm_87) 上完成 **引擎构建 + 加载 + prefill + decode + 真实文本生成**。
早前卡住推理的 FMHA 崩溃（`There must be one kernel to implement the MHA`）已定位并修复——**根因是当初在 Orin 构建 TensorRT-Edge-LLM 时漏传 `-DEMBEDDED_TARGET=jetson-orin`，导致 CMake 默认按 `sm_80;86;89` 编译并用 `-DEXCLUDE_SM_87` 把 sm_87 的 FMHA kernel 整段排除**。按官方 Orin 配方重编后即通（详见「FMHA 根因与修复」）。与量化、与模型均无关。

## 已验证跑通的链路

```
Kaggle(T4x2) 量化+导出 ONNX → 打包 tgz → 下载/校验 → scp 到 Orin → TRTEdge 建引擎 → 加载 → prefill/decode → 生成文本
   ✅ INT4/INT8           ✅        ✅ SHA256    ✅           ✅ LLM+视觉塔   ✅        ✅ 两版       ✅ 两版连贯
```

### 关键工程坑与解法（全部已解决）
| 现象 | 根因 | 解法 |
|---|---|---|
| secret HTTP 400 | API 无头运行读不到 Kaggle Notebook Secret | 脚本内嵌 HF token 回退 |
| CUDA OOM（加载） | 8B fp16(~16GB) 硬塞单块 15GB T4 | `device_map="auto"` 跨双 T4 分片加载 |
| 导出 illegal memory access | modelopt 导出假设单设备，模型却分片跨 cuda:0/1 | 导出前摘 accelerate hook、收拢到 CPU |
| `No space left`（导出/打包） | `/kaggle/working` 固定 20GB，装不下 checkpoint+onnx | 重活挪到 `/tmp`（7.9T 大盘），只把 tgz 写回 working |
| 推理崩 `There must be one kernel to implement the MHA` | **Orin 上构建 TRTEdge 漏传 `-DEMBEDDED_TARGET=jetson-orin`** → CMake 默认 `sm_80;86;89` + `-DEXCLUDE_SM_87`，sm_87 FMHA kernel 被整段排除，运行时查表必 miss | 按官方 JetPack6.2+ Orin 配方全新 build 目录重编（`-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-orin`）；详见下节 |
| bench 加载报 `Cuda Runtime (out of memory)`（int8） | bench 把 8.2GB 引擎加载两次撞 Orin 29GB 统一内存上限 | 非致命，TRT 回退统一内存后继续；功能单次加载正常，性能数稳定可复现 |
| `--multimodalEngineDir is required` | 引擎是 Qwen3VL VLM，`llm_inference` 即使纯文本也强制要视觉塔 | 传 `--multimodalEngineDir <visual 引擎目录>`（两量化共享同一份 fp16 视觉塔） |

（量化脚本：[`cloud/kaggle_cosmos_only.py`](../cloud/kaggle_cosmos_only.py)，含前三处对 TRTEdge `quantize.py` 的补丁。）

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

## 功能与性能指标（Orin, sm_87, JetPack 6, CUDA 12.6, TRT 10.7）

测试条件：batch=1，prefill inputLen=512，decode pastKVLen=512；prefill 5 iter / decode 20 iter（各含 warmup）。均用重编后的 `build_orin` 插件。

| 指标 | INT4 (AWQ) | INT8 (SmoothQuant) | 说明 |
|---|---|---|---|
| **Prefill 512 tok** | 300.98 ms | **157.45 ms** | INT8 快 ~1.9×——计算密集，原生 int8 张量核 vs int4 AWQ 反量化开销 |
| Prefill 吞吐 | 1701 tok/s | **3252 tok/s** | 同上 |
| **Decode 每 token** | **32.34 ms** | 50.65 ms | INT4 快 ~1.57×——访存密集，4.6GB vs 7.8GB 权重每 token 搬运更少 |
| Decode 吞吐 | **30.9 tok/s** | 19.7 tok/s | 交互式延迟看这项，INT4 胜 |
| 引擎载入显存 | ~4622 MiB | ~7864 MiB | Orin 统一内存 29GB；INT8 逼近上限 |
| 真实生成 | ✅ 连贯正确 | ✅ 连贯正确 | 湿路刹车/行人处置两题，两版均合理，含自然 EOS 终止 |

**结论**：交互式部署（低延迟、单流）选 **INT4**（decode 吞吐高 57%、显存省 42%）；长上下文/批量吞吐场景 INT8 prefill 更优。功能上两版输出质量均可用，无明显退化。

## FMHA 根因与修复（已解决）

**症状**：`llm_bench`/`llm_inference` 一进 context attention 就 `terminate ... what(): There must be one kernel to implement the MHA`（core dump），INT4/INT8 两版同一处、同堆栈。

**诊断链路（全部源码 + 实测坐实）**：
1. 崩在 `contextFMHARunner.cpp:458`——运行时用 9 字段 hashKey 去预编译 cubin 表精确匹配，取到空 kernel（`mSharedMemBytes==0`）。
2. 插桩打印实时 hashKey，得 `sm=87 headSize=128 mask=1(causal) tiled=0 layout=3(separate) unroll=1 fp32acc=1 flash=1 -> sharedMem=0`——请求的 key **本应命中** `fmha_cubin.h:298` 那行，却查不到。
3. 查编译宏发现 `flags.make` 带 **`-DEXCLUDE_SM_87`**，且实际 gencode 只有 `sm_80/86/89`——**sm_87 整段被排除、根本没编进 kernel 表**。插件靠 `compute_80` PTX 前向 JIT 才勉强加载到 Orin。
4. 追到 CMake：`CMakeLists.txt:64` `if(NOT DEFINED AARCH64_BUILD) set(CMAKE_CUDA_ARCHITECTURES 80;86;89)`——**当初构建漏传 `-DEMBEDDED_TARGET=jetson-orin`/toolchain，走了默认分支**，把 arch 顶成 80;86;89 并连锁排除 sm_87。

**根因一句话**：不是架构缺 cubin（sm_87 的 cubin 齐全）、不是模型不支持、也不是 hashKey 逻辑错——是**部署时的构建配置失误**（漏传 Orin target flag）。

**修复**：按官方 JetPack 6.2+ Orin 配方用全新 build 目录重编——
```bash
cd /home/vision/TensorRT-Edge-LLM && mkdir build_orin && cd build_orin
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=12.6
make -j12
```
配置期即可验证：`FMHA Kernels: Excluding SM architectures: EXCLUDE_SM_80;86;89;100;101;120;121`（**唯独保留 sm_87**）。重编后插件含 204 个 sm_87 kernel 符号，推理即通。

## 复现命令（Orin，修复后）
```bash
cd /home/vision/TensorRT-Edge-LLM
# 关键：指向重编的 sm_87 插件（插件路径由 EDGELLM_PLUGIN_PATH 决定，默认写死旧 build/）
export EDGELLM_PLUGIN_PATH=$PWD/build_orin/libNvInfer_edgellm_plugin.so
# 性能：prefill / decode
./build_orin/examples/llm/llm_bench --engineDir /home/vision/engines/int4/llm --mode prefill --inputLen 512 --iterations 5 --warmup 2
./build_orin/examples/llm/llm_bench --engineDir /home/vision/engines/int4/llm --mode decode  --pastKVLen 512 --iterations 20 --warmup 3
# 功能：真实生成（VLM 引擎需带视觉塔，即使纯文本）
./build_orin/examples/llm/llm_inference --engineDir /home/vision/engines/int4/llm \
  --multimodalEngineDir /home/vision/engines/int4/visual \
  --inputFile /home/vision/cosmos_input.json --dumpOutput --maxGenerateLength 128
```

---
name: edge-quantize-deploy
description: 把 HuggingFace LLM/VLM 用 NVIDIA TensorRT-Edge-LLM 量化(INT4 AWQ / INT8 SmoothQuant)并部署到 Jetson Orin(sm_87)跑通推理与基准测试的端到端工作流。当任务涉及 Cosmos/Qwen/Alpamayo 等模型的边缘量化、Orin 上 TRTEdge 建引擎/推理/benchmark、或排查 "There must be one kernel to implement the MHA" 类 FMHA 崩溃时使用。
---

# 边缘量化部署工作流（TensorRT-Edge-LLM → Jetson Orin）

本 skill 沉淀了 Cosmos-Reason2-8B 从云端量化到 Orin 端到端跑通的完整流程与**踩过的坑**。目标：INT4/INT8 量化 → Orin 建引擎 → prefill/decode/生成 → 指标对照。

设备约定：Orin `vision@30.245.40.99`（SSH 用 IP，别名带中文后缀会解析失败），源码 `/home/vision/TensorRT-Edge-LLM`，JetPack 6 / CUDA 12.6 / TRT 10.7 / **sm_87** / 29GB 统一内存。本地无法量化 8B（3070 仅 8GB），重活走 Kaggle 2×T4。

## 阶段 1 — 云端量化（Kaggle 2×T4=32GB，一次性）

脚本参考 `alpamayo-edge/cloud/kaggle_cosmos_only.py`。关键坑与解法：

| 坑 | 解法 |
|---|---|
| API 无头运行读不到 Notebook Secret（secret HTTP 400） | HF token 从环境变量读，**不要硬编码进脚本**（会进 git/kernel 泄露）。headless 时 `export HF_TOKEN` 后再 push |
| 8B fp16(~16GB) 塞不进单块 15GB T4（CUDA OOM） | `device_map="auto"` 跨双 T4 分片加载（PATCH A/B） |
| modelopt 导出假设单设备、模型却跨 cuda:0/1（illegal memory access） | 导出前摘 accelerate hook、把分片模型收拢到 CPU（PATCH C） |
| `/kaggle/working` 固定 20GB 装不下 checkpoint+onnx（No space left） | 重活挪 `/tmp`（大盘），只把最终 tgz 写回 working；pack 放 try/finally 防晚期失败丢产物 |

量化配置：`mtq.INT4_AWQ_CFG`（W4A16，~0.5 byte/param）/ `mtq.INT8_SMOOTHQUANT_CFG`（W8A8）。产物 = `llm/`(量化主干) + `visual/`(fp16 视觉塔，工具当前视觉塔仅 fp8/fp16) + `embedding.safetensors`(fp16)。

## 阶段 2 — 下载 + 完整性校验

- SDK `.content` 会把整个文件读进内存、无断点续传，大文件(GB级)反复断。**改用抽取签名 URL + `curl -C -` 可续传循环**下载。
- 落盘后 `sha256sum` 校验 + 存 `.sha256`；`gzip -t` 验完整性（注意大文件别套太短的 `timeout` 把 gzip -t 中途杀掉）。
- BSD/macOS `tar -tzvf` 列格式与 GNU 不同（size 不在第 3 列），提取 size 要 awk 扫字段。

## 阶段 3 — Orin 构建 TRTEdge ⚠️ 最关键、最易错

**教训（FMHA 崩溃的真根因）**：在 Orin 上构建**必须**用官方 target flag，否则 CMake 走 `CMakeLists.txt` 默认分支 `set(CMAKE_CUDA_ARCHITECTURES 80;86;89)`，连锁 `-DEXCLUDE_SM_87` 把 **sm_87 的 FMHA kernel 整段编译排除**——插件靠 compute_80 PTX 前向 JIT 勉强加载，但运行时任何 attention 查表必 miss → `There must be one kernel to implement the MHA` 崩溃。

正确配方（JetPack 6.2+ Orin，全新 build 目录）：
```bash
cd /home/vision/TensorRT-Edge-LLM && rm -rf build_orin && mkdir build_orin && cd build_orin
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=12.6
make -j12   # 全量 CUDA 重编，Orin 上 ~15-25 分钟；后台跑 + Monitor 盯 error/Built target
```
**配置期自检**（必看）：日志应打印 `FMHA Kernels: Excluding SM architectures: EXCLUDE_SM_80;86;89;100;101;120;121`——**唯独保留 sm_87** 才对。若看到排除了 87，说明 flag 没生效。重编后 `strings build_orin/libNvInfer_edgellm_plugin.so.1 | grep -c sm87_kernel_nl` 应 >0。

（`EMBEDDED_TARGET=jetson-orin` 在 toolchain 里 `set(CMAKE_CUDA_ARCHITECTURES 87)` 且 `set(AARCH64_BUILD TRUE)` 跳过默认覆盖，并加 `TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT`。CuTe DSL 非本模型必需，无 sm_87 artifact 时**省略 `-DENABLE_CUTE_DSL`**。）

## 阶段 4 — 建引擎

```bash
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
cd /home/vision/TensorRT-Edge-LLM
./build_orin/examples/llm/llm_build        --onnxDir <onnx>/llm    --engineDir <engines>/llm
./build_orin/examples/multimodal/visual_build --onnxDir <onnx>/visual --engineDir <engines>   # 输出到 <engines>/visual/
```
引擎 ~90s。两种量化可共享同一份 fp16 视觉塔引擎（视觉塔不随量化变）。

## 阶段 5 — 运行推理 + 基准

⚠️ **插件路径由 env `EDGELLM_PLUGIN_PATH` 决定，默认写死旧 `build/`**——`LD_LIBRARY_PATH` 无效。跑任何东西前必须 `export EDGELLM_PLUGIN_PATH=.../build_orin/libNvInfer_edgellm_plugin.so`。

功能（真实生成）：
```bash
./build_orin/examples/llm/llm_inference --engineDir <engines>/llm \
  --multimodalEngineDir <engines>/visual \      # VLM 引擎即使纯文本也强制要视觉塔
  --inputFile input.json --dumpOutput --maxGenerateLength 128
```
输入 JSON 格式：`{"batch_size":1,"temperature":0.7,"top_p":0.9,"top_k":50,"max_generate_length":128,"requests":[{"messages":[{"role":"user","content":"..."}]}]}`

性能：
```bash
./build_orin/examples/llm/llm_bench --engineDir <engines>/llm --mode prefill --inputLen 512 --iterations 5 --warmup 2
./build_orin/examples/llm/llm_bench --engineDir <engines>/llm --mode decode  --pastKVLen 512 --iterations 20 --warmup 3
```
读 `Prefill/Decode E2E Time` 与 `Tokens/sec`。int8(8.2GB) bench 会报**非致命** `Cuda Runtime (out of memory)`（引擎被加载两次撞 29GB 统一内存），回退后继续，数值仍稳定可复现。

功耗：另开 shell 后台 `tegrastats --interval 500 > power.log &`，跑 bench，再解析 `GPU`/`VDD_*` 列取均值，算能效 tok/J。

## FMHA / kernel 派发排查方法论（可复用）

当遇到 `There must be one kernel to implement the MHA`：
1. 崩点在 `cpp/kernels/contextAttentionKernels/contextFMHARunner.cpp` 的 `check(kernelInfo.mSharedMemBytes != 0, ...)`——运行时用 9 字段 hashKey 去预编译 cubin 表精确匹配，取到空 kernel。
2. **先查编译宏**（最快命中根因）：`grep EXCLUDE_SM build/cpp/CMakeFiles/edgellmKernels.dir/flags.make` 和实际 gencode。若 sm_87 被 EXCLUDE 或 gencode 无 sm_87 → 阶段 3 构建配置错，按上面重编。
3. 若确需看运行时请求的 key：在 `findKernelFunction(hashKey)` 后 `check` 前插 `fprintf(stderr, "[DIAG] sm=%d headSize=%d mask=%d tiled=%d layout=%d -> sharedMem=%u\n", ...)`（`fflush` 防崩前丢失），增量 `make` 重编，跑一次对照 `fmha_cubin.h` 的 `sMhaKernelMetaInfosV2[]` 行。
4. cubin 表按 SM 用 `#ifndef EXCLUDE_SM_xx` 守卫；kernel 命名 `fmha_v2_flash_attention_fp16_fp32_<Qtile>_<KVtile>_S_q_k_v_<headDim>_sm<arch>`；mask int：padding=0/causal=1/sliding=2/custom=3；layout int：packed=0/paged=2/separate=3。

## 通用纪律

- 长命令（build/download）后台跑 + Monitor 盯**成功和失败两类信号**（`Built target`/`error:`/`OOM`），别只盯成功。
- 一次性预判所有可能失败点（完整性、size、内存、路径），别逐个错误反应式处理。
- 推代码前扫明文凭据（`hf_...`/AKIA/BEGIN/password），token 只走环境变量。
- 详细结果与指标见 `alpamayo-edge/docs/edge_deploy_status.md`。

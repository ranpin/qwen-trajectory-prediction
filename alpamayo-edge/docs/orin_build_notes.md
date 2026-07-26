# Orin 构建与排障工程日志

> Cosmos-Reason2-8B 在 Jetson Orin 上从建引擎到推理跑通的工程细节：正确构建配方、
> 全部踩坑与解法、FMHA 崩溃根因深挖、复现命令。结果指标见 [edge_deploy_status.md](edge_deploy_status.md)；
> 可复用的通用工作流见仓库 skill `edge-quantize-deploy`（`.claude/skills/`）。

设备：Orin `vision@30.245.40.99`（SSH 用 IP，别名带中文后缀解析失败）。源码 `/home/vision/TensorRT-Edge-LLM`。
JetPack 6 / L4T R36.4.4 / CUDA 12.6 / TRT 10.7 / **sm_87** / 29GB 统一内存。

## 正确的 Orin 构建配方 ⚠️

**必须**用官方 target flag，否则触发下文的 FMHA 崩溃（见「FMHA 根因」）：

```bash
cd /home/vision/TensorRT-Edge-LLM && mkdir build_orin && cd build_orin
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=12.6
make -j12   # 全量 CUDA 重编，Orin 上约 15-25 分钟
```

**配置期自检**（务必看这行）：日志应打印
`FMHA Kernels: Excluding SM architectures: EXCLUDE_SM_80;86;89;100;101;120;121`
——**唯独保留 sm_87** 才对。若看到 87 被排除，说明 flag 没生效、构建会是坏的。
重编后 `strings build_orin/libNvInfer_edgellm_plugin.so.1 | grep -c sm87_kernel_nl` 应 >0（本项目为 204）。

（`-DENABLE_CUTE_DSL` 非本模型必需，且无 sm_87 artifact 时会卡，**省略**。）

## 建引擎 + 运行（复现命令）

> **插件路径机制**：TRTEdge 从 env `EDGELLM_PLUGIN_PATH` 加载插件，未设时默认相对路径
> `build/libNvInfer_edgellm_plugin.so`（`LD_LIBRARY_PATH` 无效）。本 Orin 已做运维收尾——
> 删掉旧的坏 `build/`（sm_80;86;89 + 调试插桩）、`build -> build_orin` 软链、并在 `~/.bashrc`
> 持久化绝对 `EDGELLM_PLUGIN_PATH`。故现在**默认即正确插件，export 可省**；换机器/新克隆仍需按下面显式指定。

```bash
cd /home/vision/TensorRT-Edge-LLM
# 本机已默认正确（软链+bashrc）；换环境时显式指定：
export EDGELLM_PLUGIN_PATH=$PWD/build_orin/libNvInfer_edgellm_plugin.so

# 建引擎（~90s；两量化可共享同一份 fp16 视觉编码器引擎）
./build_orin/examples/llm/llm_build            --onnxDir <onnx>/llm    --engineDir <engines>/llm
./build_orin/examples/multimodal/visual_build  --onnxDir <onnx>/visual --engineDir <engines>   # 输出到 <engines>/visual/

# 性能：prefill / decode
./build_orin/examples/llm/llm_bench --engineDir /home/vision/engines/int4/llm --mode prefill --inputLen 512 --iterations 5  --warmup 2
./build_orin/examples/llm/llm_bench --engineDir /home/vision/engines/int4/llm --mode decode  --pastKVLen 512 --iterations 20 --warmup 3

# 功能：真实生成（VLM 引擎需带视觉编码器，即使纯文本；VLM 图像路径还额外强制 --outputFile）
./build_orin/examples/llm/llm_inference --engineDir /home/vision/engines/int4/llm \
  --multimodalEngineDir /home/vision/engines/int4/visual \
  --inputFile /home/vision/cosmos_input.json --dumpOutput --maxGenerateLength 128
```

功耗测量：另开 shell `tegrastats --interval 250 > power.log &`，跑 bench，解析 GPU-active 采样的
`VDD_GPU_SOC`+`VDD_CPU_CV`+`VIN_SYS_5V0` 三轨和 = 整机功耗，除以 tok/s 得能效。

## 引擎 / 模型规格（从 Orin 引擎 config 读出）

- 架构：**Qwen3-VL 系**（Cosmos-Reason2 主干 = 后训练自 Qwen3-VL-8B-Instruct），`hiddenSize=4096`、`numDecoderLayers=36`、
  `numKVHeads=8`（GQA）、`headDim=128`、`vocabSize=151936`。
- 引擎构建参数：`maxInputLen=1024`、`maxKVCapacity=4096`、`maxBatch=4`。
- 加载：81 个 I/O 张量；含 3 路 deepstack 视觉嵌入输入（Qwen3VL）。

## 全部踩坑与解法

### 云端量化（Kaggle 2×T4）
| 现象 | 根因 | 解法 |
|---|---|---|
| secret HTTP 400 | API 无头运行读不到 Kaggle Notebook Secret | HF token 走环境变量（**勿硬编码进脚本**，会随 git/kernel 泄露） |
| CUDA OOM（加载） | 8B fp16(~16GB) 硬塞单块 15GB T4 | `device_map="auto"` 跨双 T4 分片加载（PATCH A/B） |
| 导出 illegal memory access | modelopt 导出假设单设备，模型却分片跨 cuda:0/1 | 导出前摘 accelerate hook、收拢到 CPU（PATCH C） |
| `No space left`（导出/打包） | `/kaggle/working` 固定 20GB，装不下 checkpoint+onnx | 重活挪 `/tmp`（大盘），只把 tgz 写回 working；pack 放 try/finally |

（量化脚本 [`cloud/kaggle_cosmos_only.py`](../cloud/kaggle_cosmos_only.py) 含上述三处对 TRTEdge `quantize.py` 的补丁。）

### 下载 / 部署 / 运行
| 现象 | 根因 | 解法 |
|---|---|---|
| SDK 下载反复断（GB 级） | `.content` 整文件进内存、无断点续传 | 抽签名 URL + `curl -C -` 可续传循环；SHA256 + `gzip -t` 校验 |
| **推理崩 `There must be one kernel to implement the MHA`** | **构建漏传 `-DEMBEDDED_TARGET=jetson-orin`**，sm_87 FMHA kernel 被编译排除（详见下节） | 按上面正确配方重编 |
| bench 报 `Cuda Runtime (out of memory)`（int8） | bench 把 8.2GB 引擎加载两次，撞 29GB 统一内存上限 | **非致命**：TRT 回退统一内存后继续；功能单次加载正常，性能数稳定可复现 |
| `--multimodalEngineDir is required` | 引擎是 Qwen3VL VLM，`llm_inference` 即使纯文本也强制要视觉编码器 | 传 `--multimodalEngineDir <visual 引擎目录>` |
| `--outputFile is required`（图像输入） | VLM 图像路径强制落盘输出 | 传 `--outputFile` |

## FMHA 崩溃根因深挖（已解决）

**症状**：`llm_bench`/`llm_inference` 一进 context attention 就
`terminate ... what(): There must be one kernel to implement the MHA`（core dump），INT4/INT8 两版同处、同堆栈。

**诊断链路（源码 + 实测坐实）**：
1. 崩在 `cpp/kernels/contextAttentionKernels/contextFMHARunner.cpp:458`——运行时用 9 字段 hashKey
   去预编译 cubin 表精确匹配，取到空 kernel（`mSharedMemBytes==0`）触发断言。
2. 插桩打印实时 hashKey：`sm=87 headSize=128 mask=1(causal) tiled=0 layout=3(separate) unroll=1 fp32acc=1 flash=1 -> sharedMem=0`——请求的 key **本应命中** `cubin/fmha_cubin.h:298`（`..._128_causal_sm87_kernel_nl`），却查不到。
3. 查编译宏：`flags.make` 带 **`-DEXCLUDE_SM_87`**，实际 gencode 只有 `sm_80/86/89`——**sm_87 整段被 `#ifndef EXCLUDE_SM_87` 编译排除、根本没进 kernel 表**（cubin 文件在盘上存在但未注册）。插件靠 `compute_80` PTX 前向 JIT 才勉强加载到 Orin。
4. 追到 CMake `CMakeLists.txt:64`：`if(NOT DEFINED AARCH64_BUILD) set(CMAKE_CUDA_ARCHITECTURES 80;86;89)`——**当初构建漏传 `-DEMBEDDED_TARGET=jetson-orin`/toolchain，走了默认分支**，把 arch 顶成 80;86;89 并连锁 `-DEXCLUDE_SM_87`。

**根因一句话**：不是架构缺 cubin（sm_87 的 cubin 齐全，与 sm_80/89 等量）、不是模型不支持、也不是 hashKey 逻辑错——是**部署时的构建配置失误**（漏传 Orin target flag）。与量化、模型均无关。

**修复**：按上面「正确构建配方」全新 build 目录重编；配置期排除列表翻转为"只留 sm_87"，推理即通。

**排障方法论**（遇到同类 kernel 派发崩溃可复用）：先查编译宏 `grep EXCLUDE_SM build/cpp/CMakeFiles/*/flags.make`
和 gencode（最快命中根因）→ 需要看运行时 key 再在 `findKernelFunction` 前插 `fprintf`（记得 `fflush`，崩前不丢）
→ 对照 `fmha_cubin.h` 的 `sMhaKernelMetaInfosV2[]` 行。mask int：padding=0/causal=1/sliding=2/custom=3；
layout int：packed=0/paged=2/separate=3。

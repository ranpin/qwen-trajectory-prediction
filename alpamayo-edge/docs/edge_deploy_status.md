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

### 功耗与能效（tegrastats，MAXN 模式，GPU-active 采样均值）

三条电源轨：`VDD_GPU_SOC`(GPU+SOC 主算力) + `VDD_CPU_CV`(CPU) + `VIN_SYS_5V0`(5V 外设) = 整机模块功耗。

| 场景 | INT4 (AWQ) | INT8 (SmoothQuant) | 能效赢家 |
|---|---|---|---|
| **Prefill** GPU_SOC / 整机 | 48.0 W / 61.8 W | 39.5 W / 52.1 W | — |
| Prefill 能效 | 27.5 tok/J | **62.8 tok/J** | **INT8 快 2.3×**（原生 int8 核，功耗更低+吞吐更高） |
| **Decode** GPU_SOC / 整机 | 26.3 W / 39.6 W | 25.5 W / 40.6 W | — |
| Decode 能效 | **0.82 tok/J** | 0.50 tok/J | **INT4 高 1.64×**（访存密集，权重小=搬运能耗低） |
| 峰值整机功耗 | prefill 67.3 W / decode 49.4 W | prefill 61.7 W / decode 46.6 W | — |

**结论（含能效）**：
- **交互式 / 单流低延迟 / 省电续航** → **INT4**：decode 吞吐高 57%、能效高 64%、显存省 42%。
- **批量 / 长上下文 / 高吞吐** → **INT8**：prefill 快 1.9×、能效高 2.3×、功耗还更低。
- 一句话：**decode 看 INT4，prefill 看 INT8**——两者在不同阶段各自既更快又更省电。功能上两版输出质量均可用，无明显退化。

### 多模态（图像）路径验证 ✅

喂 `examples/multimodal/pics/woman_and_dog.jpeg` + 提问，走**视觉塔 + deepstack 融合 + INT4 LLM** 全路径（`Processing vision inputs` / `Vision runner successfully initialized`）。模型准确描述了海滩场景（女子格子衫、狗戴彩色胸背带、海浪、日落暖光），与官方参照描述吻合 → VLM 本职能力在 Orin 上端到端正确。VLM 路径 CLI 额外强制 `--outputFile`。

### 吞吐 / 上下文扩展性 sweep（INT4，MAXN）

| Prefill inputLen (batch1) | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| tok/s | 1485 | 1631 | 1701 | 1720 |

> 长输入吞吐更高（利用率摊薄），1024 趋稳。

| Prefill batch (inputLen512) | 1 | 2 | 4 |
|---|---|---|---|
| per-seq tok/s | 1699 | 873 | 422 |
| 聚合 tok/s | 1699 | 1746 | 1686 |

> **prefill 在 batch=1 已算力饱和**，批处理聚合吞吐恒定 ~1700，无增益（Orin GPU 打满）。

| Decode pastKVLen (batch1) | 128 | 512 | 1024 | 2048 | 4000 |
|---|---|---|---|---|---|
| tok/s | 32.1 | 31.8 | 31.3 | 30.5 | 29.2 |

> 上下文从 128 涨到 4000（31×），decode 仅掉 9%——由权重搬运主导，注意力占比小，长上下文扩展性好。

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

## 下一步实验建议（按性价比排序）

已跑通"链路 + 双量化延迟/吞吐/显存 + 功能连贯"。要让评测更完整、更有说服力，建议：

1. **功耗测量（补齐 M4）** — 优先级最高、成本最低。decode/prefill 时后台跑 `tegrastats`（或读 `/sys/bus/i2c/.../power`），记 INT4/INT8 的 GPU+SOC 功耗(W) 与 **能效(tok/J)**。边缘部署这项最有说服力，且几分钟可得。
2. **FP16 基线补第三个点** — 现在只有 INT4/INT8 两点。补一个未量化 FP16 引擎（需在 Kaggle 再导一次 base ONNX，或查 TRTEdge 能否直接 fp16 build），才能画出完整的"精度-延迟-显存"三点权衡曲线，量化收益才量化得出来。
3. **量化掉点定量评测（补齐 M3）** — 现仅 2 条 prompt 的定性判断。建一个小评测集（数十条 AV/推理题），用 FP16 输出作参照，算 INT4/INT8 的一致性/perplexity/任务正确率，得"掉点 <Z%"的硬数字。
4. **真·多模态路径** — Cosmos-Reason2 是 VLM，目前只测了纯文本。喂真实图像 + 场景问题，走视觉塔 + deepstack 路径（模型的本职：物理场景推理）。这也验证视觉塔引擎的端到端正确性。
5. **吞吐/上下文扩展性 sweep** — prefill 扫 inputLen(128→1024) 与 batch(1→4，引擎 maxBatch=4)；decode 扫 pastKVLen(128→4096) 看延迟随上下文增长曲线。刻画真实负载下的表现。
6. **运维收尾** — 把 `build_orin` 设为默认（替换旧 `build/` 或持久化 `EDGELLM_PLUGIN_PATH`），清掉旧的带 DIAG 插桩的 `build/`；可选：向上游报"在 Orin 本机不带 `-DEMBEDDED_TARGET` 构建会静默排除 sm_87 导致 FMHA 崩"的体验问题。
7. **（大工程）VLA 轨道 M2b** — 导出 Alpamayo-R1-10B(FP16 + action 头) → Orin build → 出轨迹 → 接 `eval/` 的 ADE/FDE/MR vs CVM。这是原计划的第二条轨道，独立且工作量大。

> 建议顺序：先 **1+2+3**（补全量化评测三件套，让"精度-延迟-功耗"权衡完整）→ 再 **4**（多模态本职能力）→ 视精力做 **5/6/7**。

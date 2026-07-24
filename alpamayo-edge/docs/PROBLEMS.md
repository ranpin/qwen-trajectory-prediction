# 问题记录 —— 全流程踩坑与解决（统一）

> 本项目实验过程中遇到的**所有**问题、根因与解决方案，集中在此一处。呈现文档（主页面）不含这些细节，只在此归档。
> 每条：**现象 → 根因 → 解决 → 教训**。均为真实发生、可复现。更新：2026-07-22。

## A. 云端量化 / 导出（Kaggle）

### A1. 8B 加载 CUDA OOM
- **现象**：单卡加载 `Cosmos-Reason2-8B` 直接 OOM。
- **根因**：8B fp16≈16GB，单张 T4(16GB) 装不下权重+激活。
- **解决**：`device_map="auto"` 把模型分片到 2×T4；导出前用 `accelerate.remove_hook` + `model.to("cpu")` 把分片模型合并回 CPU 再 export（quantize.py PATCH A/B/C）。
- **教训**：跨多卡分片加载后，导出前必须先合并到单一设备，否则 ONNX 导出会因跨设备张量报错。

### A2. `No space left on device`
- **现象**：int8 检查点(~10GB)+ONNX(~10GB) 撑爆 `/kaggle/working`。
- **根因**：`/kaggle/working` 是固定 20GB loop 设备。
- **解决**：所有重活写 `/tmp`（挂在 7.9TB overlay `/` 上，1.1TB 空闲）；导出后立刻删检查点，只把最终 tgz 放回 `/kaggle/working`；打包放 `try/finally` 确保晚期失败也不丢已产出物。
- **教训**：认清云环境各挂载点的真实容量，重 IO 一律走大盘。

### A3. `secret HTTP 400` —— API 跑读不到 Notebook Secret
- **现象**：`kaggle kernels push` 触发的后台运行读不到 HF_TOKEN Secret。
- **根因**：API 触发的 run 无法读取 UI 配的 Notebook Secret。
- **解决**：推送的私有 kernel 临时副本里注入 token（env 方式）；**repo 副本保持无 token**。
- **教训**：headless 跑云 kernel 时凭据走 env 注入，且注入副本与版本库副本分离。

### A4. HF 门控 401 / `GatedRepoError`
- **现象**：加载 `nvidia/Cosmos-Reason2-8B` 报 401，t≈47s 快速失败（未浪费 GPU）。
- **根因**：本地默认 token（`~/.cache/huggingface/token`）**未获该门控仓库授权**。
- **解决**：改用已接受许可、有 Cosmos 访问权的 token。
- **教训**：门控模型先确认 token 的仓库授权；鉴权失败会很快暴露，把它当作廉价的前置检查。

### A5. Kaggle GPU 抽签给了 P100，torch 不兼容
- **现象**：FP16 参考跑 `generate()` 崩 `CUDA error: no kernel image is available`（`cudaErrorNoKernelImageForDevice`）。
- **根因**：API 默认 `enable_gpu:true` 给的是**单张 Tesla P100（sm_60）**；Kaggle 预装的新版 PyTorch 只支持 `sm_70..sm_120`，不含 sm_60。
- **解决**：kernel-metadata 加 `"machine_shape":"NvidiaTeslaT4"` 锁定 **T4×2（sm_75）**。
- **教训**：API 推送**默认不给 T4×2**，必须显式指定 `machine_shape`；否则 GPU 类型随机、可能撞上不受支持的架构。

## B. Orin 构建

### B1.（最硬）FMHA 崩溃 `There must be one kernel to implement the MHA`
- **现象**：推理一进 attention 即崩。
- **根因**：Orin 构建 TensorRT-Edge-LLM **漏传 `-DEMBEDDED_TARGET=jetson-orin`**，CMake 走默认分支按 `sm_80;86;89` 编译并 `-DEXCLUDE_SM_87`，把 **sm_87 的 FMHA kernel 整段编译排除**；插件靠 compute_80 PTX 前向 JIT 才勉强加载，但运行时 attention 查表必落空。**非架构缺 cubin、非模型、非量化。**
- **解决**：按官方 Orin 配方全新重编（`-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=12.6`），配置期 FMHA 排除列表翻转为"只留 sm_87"。
- **教训**：交叉编译/嵌入式目标，构建 flag 缺失会**静默**产出能加载但运行必崩的产物；排查要跨层（运行时 kernel 派发表 ↔ 预编译 cubin ↔ 编译宏 ↔ CMake）。深挖全过程见 [orin_build_notes.md](orin_build_notes.md)。

### B2. 插件路径不受 `LD_LIBRARY_PATH` 控制
- **现象**：换了新 build 目录，推理仍加载旧插件（带旧 sm 列表）。
- **根因**：插件由环境变量 `EDGELLM_PLUGIN_PATH` 决定（默认相对路径 `build/libNvInfer_edgellm_plugin.so`），与 `LD_LIBRARY_PATH` 无关。
- **解决**：`export EDGELLM_PLUGIN_PATH=<绝对路径>`；并 `build→build_orin` 软链 + `~/.bashrc` 持久化。
- **教训**：先搞清框架到底用哪条路径加载插件，别想当然靠 `LD_LIBRARY_PATH`。

## C. Orin 推理 / 评测

### C1. `--multimodalEngineDir is required`
- **现象**：纯文本推理也报错要视觉塔。
- **根因**：VLM 引擎即使只跑文本也要求带多模态引擎目录。
- **解决**：`--multimodalEngineDir <int4/visual>`（int8 复用 int4 的 fp16 视觉塔）；VLM 路径 CLI 还需 `--outputFile`。
- **教训**：VLM 的文本推理仍需完整多模态引擎在位。

### C2. llm_bench 双加载逼近 29GB
- **现象**：int8 bench 出现非致命 OOM 告警。
- **根因**：引擎被加载两次，撞 29GB 统一内存；TRT 回退到统一内存，数字仍稳定。
- **风险外推**：FP16 引擎(~16GB) 若双加载→32GB 必爆。
- **解决/对策**：FP16 性能测量改用 **llm_inference 单次加载 + `--dumpProfile`**，避开双加载。
- **教训**：大引擎测性能前，先确认工具的加载次数与显存峰值。

## D. 部署 / 呈现

### D1. 提交里混入明文 HF token
- **现象**：`cloud/kaggle_cosmos_only.py` 内嵌明文 token。
- **解决**：改纯 env 读取、无硬编码回退，amend 后再推 → **token 从未进过 GitHub 历史**。
- **教训**：推送前做全历史密钥自查；凭据只走 env。

### D2. 主站外链 404（`/edge-ai-docs/https://…`）
- **现象**：跨子站项目卡片点击 404。
- **根因**：主站 `DocsSection` 无条件给 `file` 拼 `DOCS_BASE` 前缀。
- **解决**：`DocCard` 判断 `file` 以 `http(s)://`/`//` 开头则原样用（PR #24）。
- **教训**：目录清单要原生支持相对条目与绝对外链两种。

## E. 基准评测（Cosmos-Reason1-Benchmark）

### E1. `kaggle kernels output` 下载 15GB 大文件卡死在 0B
- **现象**：拉云端 FP16 ONNX(15GB) 时进程挂起，文件停在 0B。
- **根因**：kaggle CLI 的 `kernels_output` 用 `requests.get(url, stream=True)` 却又 `out.write(resp.content)`——**把整个 15GB 读进内存**再写盘，内存扛不住。
- **解决**：用 API 取签名 URL，`curl -L --http1.1 -C -` **流式+断点续传**（服务器还会 HTTP/2 断流，故 `--http1.1` + 循环重试直到 `tar tzf` 校验通过）。
- **教训**：大文件别用会全量读内存的下载器；流式+可续传+完整性校验。

### E2. 基准仓库结构非 parquet
- **现象**：按 datasets-server 显示的 parquet 列去读，`No objects to concatenate`。
- **根因**：datasets-server 的 parquet 是自动转换的；**真实仓库是 `*_qa_pairs.json`（标注）+ `clips.tar.gz`（视频，需解包）**。
- **解决**：从 JSON 读记录、解包 clips.tar.gz、按相对路径 `clips/xxx.mp4` 定位。
- **教训**：先看仓库真实文件布局（snapshot 后 `ls`），别只信 viewer。

### E3. 视觉引擎 patch 上限 —— 多帧超限
- **现象**：6 帧全分辨率多图推理，`qwenViTRunner: cuSeqlens 6120 exceeds maxHW=4096`，全部请求失败。
- **根因**：Orin 视觉(ViT)引擎构建上限 **总 patch 数 4096**；6 帧 × ~1020 patch = 6120 超限。
- **解决**：帧**缩放到 ≤448px**（每帧 ~256 patch，6 帧≈1536<4096）；**FP16 与 Orin 用同一批缩放帧**保证掉点可比。
- **教训**：边缘视觉引擎有固定输入上限；多图/多帧要按 patch 预算设计分辨率与帧数。

### E4. FP16 引擎在 Orin build 期 OOM
- **现象**：`llm_build` FP16 引擎，进程消失、引擎目录空、detached 重试 **exit 137（SIGKILL/OOM）**。
- **根因**：FP16 权重 ~16GB + TensorRT builder workspace 峰值 > 29GB 统一内存（lean `--maxKVCacheCapacity` 只降运行时 KV，不降 build 峰值）。
- **实测根因（2026-07-24 用 64GB goat 验证）**：build 峰值 **54.8GB**（>30GB），运行时激活仅 8MB → 纯 **build 内存墙**，非推理。**解法**：在 64GB orin-goat build（成功，peak 54.8GB），引擎同 sm_87+TRT10.7 可拷回 32GB dog 跑（推理 ~17GB）→ 得到 FP16 基线与量化加速比（decode INT4 2.66×）。
- **教训**：边缘设备上大模型 FP16 常连 build 都过不去；量化不只是提速，而是**能否部署**的前提。

---
> 说明：**量化校准集用的是通用新闻文本（非驾驶域）** 属于**方法局限/口径**而非 bug，其影响分析见 [METHODOLOGY.md](METHODOLOGY.md) §4 与主页面「思考讨论」。

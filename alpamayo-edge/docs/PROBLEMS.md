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


### F1. 换工作目录后引擎"反序列化失败"（2026-07-25）

- **现象**：`llm_inference` 报 `failed to deserialize engine: .../llm.engine`，而同一引擎用 `llm_bench` 一直正常。
- **根因**：插件路径。`~/.bashrc` 开头对**非交互式 shell 会提前 `return`**，所以 ssh 里 `source ~/.bashrc` 并不会设上 `EDGELLM_PLUGIN_PATH`；此前 `llm_bench` 能跑纯属侥幸——命令的 cwd 在 `/home/vision/TensorRT-Edge-LLM/` 下，命中了默认的**相对**路径 `build/libNvInfer_edgellm_plugin.so`（`build` 是指向 `build_orin` 的软链）。这次把 cwd 换到 `/home/vision` 后相对路径失效 → 自定义插件未注册 → 引擎反序列化必失败。
- **解决**：非交互式调用一律**显式** `export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so`。
- **教训**：报错信息（反序列化失败）与真因（插件没加载）相隔很远；"换个目录就崩"要先怀疑相对路径依赖。同理 `nvcc` 也不在非交互式 ssh 的 PATH 里，需 `export PATH=/usr/local/cuda/bin:$PATH`。


### G1. 上游源码漂移让 Kaggle 量化 run 直接失败（2026-07-26）

- **现象**：新推的 `alpamayo-edge-lmhead` kernel 跑到 169 s 就 `ERROR`，日志末尾：
  `AssertionError: PATCH C needle not found — TRTEdge source changed upstream`（`/kaggle/src/script.py:116`）。
- **根因**：脚本用 `git clone --depth 1`（**未 pin commit**）拉 TensorRT-Edge-LLM，再对
  `quantization/quantize.py` 打三处字符串补丁。上游在 `os.makedirs(output_dir, exist_ok=True)` 与
  `with torch.inference_mode(), _skip_resmooth_for_hybrid(` **之间插入了一段注释**，
  于是要求两行相邻的 PATCH C needle 失配。PATCH A/B 仍匹配。
- **解决**：把 PATCH C 的锚点改成**只锚单行** `    os.makedirs(output_dir, exist_ok=True)`（在当前上游文件中仍唯一，已核对 `grep -c` = 1），
  插入位置改为该行之前。同时确认**上游至今仍未自己做 sharded→CPU 合并**（grep `remove_hook_from_module` / `to("cpu")` 均无命中），
  所以 PATCH C 依然必要。修复已回写到 `cloud/kaggle_lmhead_int4.py` **和**原始的 `cloud/kaggle_cosmos_only.py`。
- **教训**：
  1. **这次 assert 起了正作用**——它让失败在 169 s 内明确暴露，而不是"看似打了补丁"后在几十分钟的量化尾声炸掉、或更糟：**静默产出一个错的模型**。字符串补丁必须配唯一性断言。
  2. **锚点要选最小且稳定的片段**。要求多行相邻＝把上游的排版当契约，注释一改就断。
  3. **`clone --depth 1` 不 pin commit 是可复现性缺口**（§4.1 已要求如实标注）。真正的修法是 pin 到 commit；
     当前保持 main 以便拿上游修复，代价就是这类漂移，已在此明确记录。
  4. 失败 run 的 `/kaggle/working` 残留（含整个 TRTEdge 源码树）可通过 `kaggle kernels output` 拉回，
     **正好用来直接读当前上游源码定位漂移**——比盲猜快得多。日志要用 `kaggle kernels logs`（`kernels output` 不含日志）。


### G2. 量化 lm_head 时 T4 CUDA OOM（2026-07-26）

- **现象**：`alpamayo-edge-lmhead` v2 跑到 845 s 失败：
  `tensorrt-edgellm-quantize ... --lm_head_quantization int4_awq` 退出码 1，
  栈底是 `modelopt/torch/quantization/tensor_quant.py:_quantize_impl` →
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB.
  GPU 1 has a total capacity of 14.56 GiB of which 1006.81 MiB is free`。
- **根因**：`device_map="auto"` 把 8B(fp16≈16GB) 尽量塞满两张 T4（每张 14.56 GiB，实际用到 13.58 GiB），
  **不给量化算子留临时空间**。而 lm_head 是 `4096 × 151936 = 622 M` 参数，
  fake-quantize 需要约 **1.16 GiB** 的临时张量（与报错数值吻合）——**恰恰是骨干各层都不需要的那种大块临时内存**，
  所以骨干 36 层全部量化成功、只在最后的 lm_head 上炸。
- **解决**：给 PATCH A 的 `from_pretrained` 加
  `max_memory={0: "11GiB", 1: "11GiB", "cpu": "60GiB"}`，
  每张卡留出 ~3.5 GiB 余量给量化临时张量；11+11=22 GiB 仍足以把 16 GB 模型全放在 GPU 上（不牺牲速度）。
- **教训**：
  1. **"模型能装下"≠"能量化"**——量化过程本身需要额外的峰值内存，且**与被量化张量的最大单层尺寸成正比**。
     这与之前 FP16 引擎的 TRT build 内存墙（权重×~3）是同一类错误的不同版本：
     **算显存只算权重，必然低估**。
  2. 词表投影层（vocab × hidden）通常是整个模型**单层最大**的权重，任何"逐层处理"的流程
     （量化 / 导出 / 建引擎）都最可能在它上面爆——优化它收益也最大（见 K1），风险也最集中。
  3. `device_map="auto"` 的默认目标是"尽量塞满"，凡是后续还要做重活的流程都应显式给 `max_memory` 留头寸。


### G3. 在 2×T4 上量化 8B 的 lm_head：五次实验证明是资源死路（2026-07-26）

**目标**：K1 剖析发现 lm_head 未量化、独占 decode 字节 25.6%，预测量化它可得 +23% decode。
TRTEdge 有现成开关 `--lm_head_quantization int4_awq`，看起来只是加个参数。**实际做了 5 次都没成。**

| run | max_memory | 实测 placement | 失败模式 |
|---|---|---|---|
| v2 | 未设 | 全 GPU | `PATCH C needle not found`（上游漂移，见 G1） |
| v3 | 11 GiB/卡 | 全 GPU | **CUDA OOM**：lm_head fake-quantize 要 1.16 GiB，GPU1 仅剩 1006.81 MiB |
| v4 | 9 GiB/卡 | `{0:16, 1:24, cpu:1}` | **CPU offload → AWQ 校准 `illegal memory access`** |
| v5 | 10 GiB/卡 | `{0:16, 1:25}` 全 GPU | **CUDA OOM，剩余显存与 v3 完全相同（1006.81 MiB）** |
| v6 | 11/6 非对称 | `{0:16, 1:16, cpu:9}` | **同 v4 崩溃** |

**根因（两面夹击，在 14.56 GiB×2 上无解）**：
1. 上限够高到把所有模块留在 GPU ⇒ 承载 lm_head 的那张卡剩不下 1.16 GiB 临时空间 ⇒ **OOM**。
2. 上限低到腾出空间 ⇒ 模块外溢到 CPU ⇒ **ModelOpt 的 AWQ 校准与 accelerate 的 offload hook 不兼容**，
   在 `F.linear` 处 illegal memory access ⇒ 直接崩。
3. **关键认识：`max_memory` 只约束"初始权重放置"**，量化状态之后照样把显存吃到同一水位——
   v3(11 GiB) 与 v5(10 GiB) OOM 时**剩余显存一模一样（1006.81 MiB）**，这就是铁证。
   所以"压总量"根本不是有效杠杆。

**出路**：需要**单卡 ≥24 GB**（L4 / A6000 / A100），让 8B + 1.16 GiB 临时张量无需分片即可容纳，
从而同时避开两个失败模式。仓库已有 `cloud/colab_export_quant.ipynb`（Colab 可给 L4/A100），
但 Colab 需要交互式会话。**+23% 至今仍是"预测未验证"，文档中一律如此标注，不得写成已达成。**

**教训**：
1. **"只是加一个 CLI 参数"可能触发完全不同的资源画像**。lm_head 是 622 M 参数、
   骨干最大单层（50 M）的 **12 倍**，逐层处理的流程必然在它上面先爆——
   与 G2 是同一条规律的两个实例。
2. **诊断要打印实际状态而不是相信配置**：加了一行 `hf_device_map` 汇总打印，
   才发现 v4/v6 静默外溢到 CPU；否则只会看到"又崩了"。
3. **同样的错误码不代表同样的原因**（v3/v5 是 OOM，v4/v6 是 illegal memory access），
   但**完全相同的剩余显存数值**反而揭示了"cap 无效"这个更深的机制。异常里的数值巧合值得追。
4. 失败要**便宜**：每次 run 都在 15 分钟内失败并留下可读日志，5 次总成本仍小于一次盲目的长跑。

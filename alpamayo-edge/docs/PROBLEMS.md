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
- **现象**：纯文本推理也报错要视觉编码器。
- **根因**：VLM 引擎即使只跑文本也要求带多模态引擎目录。
- **解决**：`--multimodalEngineDir <int4/visual>`（int8 复用 int4 的 fp16 视觉编码器）；VLM 路径 CLI 还需 `--outputFile`。
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
- **解决**：帧**长边缩到 448px**；实测（2026-07-26 复核）每帧 patch 数为 **448**（448×252→448×256，28×16）或 **616**（448×358→448×352，28×22），6 帧 = 2688 / 3696 < 4096 ⇒ 通过，但 5:4 那批已用掉 90% 的 patch 预算。**FP16 与 Orin 用同一批缩放帧**保证精度损失可比。（此前本条写的"每帧 ~256 patch"是估算，实测已修正。）
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


### G3. 在 2×T4 上量化 8B 的 lm_head：五次实验证明是资源限制导致的不可行（2026-07-26）

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

#### G3 续（2026-07-27，v7/v8：两个新失败模式，终于定位到真正的机制）

又试了两次（kernel `alpamayo-edge-lmhead-2b8b`，557 s 内双双失败，日志已归档）：

| run | 配置 | 实测 placement | 失败模式 |
|---|---|---|---|
| v7 | **2B**，`device_map={"": 0}` 单卡 | 全在 GPU0 | **CUDA OOM：要分配 4.64 GiB**，GPU0 仅剩 1.94 GiB（进程已占 12.62 GiB） |
| v8 | **8B**，`max_memory={0:"13GiB", 1:"7GiB"}` 非对称 | **`{0: 16, 1: 19, disk: 6}`** | **6 个模块被 offload 到磁盘** → 同 v4/v6 的 `illegal memory access` |

**两个假设都被自己的实测推翻：**
1. 我以为"非对称且总和 20 GiB > 模型 16.34 GiB 就不会外溢"——**错**。accelerate 每卡还要留激活/账务头寸，
   20 GiB 的 `max_memory` 装不下 16.34 GiB 的模型，**而且这次外溢的不是 CPU 而是 `disk`**。
   任何形式的 offload（CPU 或磁盘）都会让 AWQ 校准挂在 `F.linear`。
2. 我以为"2B fp16 才 4.3 GiB，单卡绝对够"——**错**。量化流程的进程峰值是权重的约 **3 倍**（4.3 → 12.62 GiB）。

**但 v7 的报错数字第一次指向了真正的机制。** 4.64 GiB 是什么？
`校准 batch 16 × seq 512 × vocab 151936 × 4 B(fp32) = 4.64 GiB` —— **与报错数值精确吻合**。
⇒ **量化 lm_head 需要为校准批次materialize 完整的 logits 张量，它的大小由 `batch × seq × vocab` 决定，
与权重大小无关。** 这解释了为什么"换更小的模型"救不了：2B 和 8B 的 vocab 都是 151936，
**logits 张量一样大**，缩小模型完全没有减少这一项。
（对比 8B 早前 v3/v5 报的 1.16 GiB = `4096×151936×2 B`，那是 lm_head **权重**本身——是另一个张量。
两个张量都致命，但机制不同。）

**因此正确的出路多了一条、而且不需要更大的显卡**：把 **AWQ 校准 batch 从 16 降到 1–2**
（§4 记录的默认值是 AWQ=16），logits 张量随之缩小 8–16×（4.64 GiB → 0.29 GiB）。
这比"等一张 ≥24 GB 的卡"便宜得多，是下一次该试的第一件事。

#### G3 结局（2026-07-27，v9：**成功**，第 9 次）

按上面的机制改了一处（`PATCH D`，`quantization/quantize.py:737`）：

```diff
-            batch_size = 16 if quantization in (None, "int4_awq") else 1
+            batch_size = 1
```

同时把 8B 的 `max_memory` 从 v8 的 13/7 改回**对称 12/12**（v8 已证 13/7 会把 6 个模块外溢到磁盘；
batch=1 后临时张量小一个数量级，不必再压总量）。

**结果：2B 与 8B 双双成功。**

| run | 配置 | 实测 placement | 结果 |
|---|---|---|---|
| v9-2B | `device_map={"": 0}` + calib batch 1 | 全 GPU0 | ✅ `COSMOS_2b_int4_awq_lmh_OK` |
| v9-8B | `max_memory={0:"12GiB",1:"12GiB"}` + calib batch 1 | **`{0: 16, 1: 25}`** —— 全在 GPU，**无 CPU、无 disk** | ✅ `COSMOS_8b_int4_awq_lmh_OK` |

产物 `edge_artifacts_lmhead.tgz`（6.5 GB，含两个模型的 ONNX），`DONE ok=True`。
**代价：全程 190 分钟**（v7/v8 失败只用了 9 分钟）——batch 16→1 让 512 条样本逐条过，校准慢约 12×。

**教训（这条是整个 G3 最值钱的部分）**：
1. **报错里的数字要做因式分解，而不是只读"又 OOM 了"。** 前 8 次尝试全在调"显存总量"和"模块放到哪张卡"
   这两个维度上打转，而真正的自变量藏在 `4.64 GiB = 16 × 512 × 151936 × 4 B` 里——**是校准批大小**。
   一旦看对了自变量，**第一次尝试就两个模型全过**。
2. **"换个更小的模型"不是万能降级路径**：瓶颈项 `batch × seq × vocab` 与模型规模无关，
   2B 和 8B 的 vocab 都是 151936 ⇒ 8B→2B 对这一项毫无帮助（v7 的 2B 单卡失败正是这么来的）。
   降级之前先确认瓶颈随什么变化。
3. **失败要便宜**：8 次失败每次 9–15 分钟就明确退出，总成本远小于一次盲目的长跑；
   而唯一成功的那次才花 190 分钟。**先便宜地把机制搞对，再付昂贵的算力。**

**⚠️ 口径变更，未验证收益前不得改写结论**：校准 batch 会影响 AWQ 的 per-channel scale 估计，
所以 **batch=1 量出来的模型与已部署的 batch=16 版 INT4 不是同一个量化配置**（样本数仍是 512，覆盖度不降）。
+19~23% 的 decode 增益**仍未验证**——还需要：① 产物拉回本地 ② 传到 Orin 建引擎
③ `llm_bench` 测 decode ④ **同时**在 n=210 基准上测正确率，与 batch=16 版并列报告。
只报速度不报精度，等于拿一个换了配方的模型宣称优化生效。

**教训**：
1. **报错里的数字要拿去做因式分解。** 5 次失败只看到"又 OOM 了"；第 7 次把 4.64 GiB 拆成
   `batch×seq×vocab×4B` 才看见真正的自变量是**校准批大小**，而不是显存总量或模块放置。
2. **`max_memory` 的语义比想象的弱**：它既不约束量化期的峰值（v3/v5 已证），也不保证"总和够就不外溢"（v8 新证），
   而且外溢目标可能是**磁盘**而非 CPU。**必须打印 placement 才知道真相**——这次正是那行打印暴露了 `disk: 6`。
3. **"换个更小的模型"不是万能的降级路径**：当瓶颈项与模型规模无关（这里是 `vocab`），缩小模型毫无帮助。
   降级前先确认瓶颈到底随什么变化。

**教训**：
1. **"只是加一个 CLI 参数"可能触发完全不同的资源画像**。lm_head 是 622 M 参数、
   骨干最大单层（50 M）的 **12 倍**，逐层处理的流程必然在它上面先爆——
   与 G2 是同一条规律的两个实例。
2. **诊断要打印实际状态而不是相信配置**：加了一行 `hf_device_map` 汇总打印，
   才发现 v4/v6 静默外溢到 CPU；否则只会看到"又崩了"。
3. **同样的错误码不代表同样的原因**（v3/v5 是 OOM，v4/v6 是 illegal memory access），
   但**完全相同的剩余显存数值**反而揭示了"cap 无效"这个更深的机制。异常里的数值巧合值得追。
4. 失败要**便宜**：每次 run 都在 15 分钟内失败并留下可读日志，5 次总成本仍小于一次盲目的长跑。

## H2. 抬高 maxInputLen 会静默让 INT8 的 prefill 慢 15%（2026-07-27）

- **现象**：把 `maxInputLen` 从默认 1024 提到 4096 重建引擎后，**INT8 在 512/1024 处的 prefill 反而慢了
  15.3% / 15.5%**（157.66 → 181.78 ms、312.42 → 360.86 ms），而 **INT4 完全不受影响**（≤0.5%）。
- **根因**：`cpp/builder/llmBuilder.cpp` 把 TensorRT 优化 profile 写成
  `optCtxShape = {maxBatchSize, maxInputLen/2, hidden}`、`maxCtxShape = {maxBatchSize, maxInputLen, hidden}`
  （变量名自证）⇒ **opt 形状被硬编码为 max 的一半**。抬高上限就把 TRT 的调优点从 512 挪到 2048，
  典型长度反而离最优点更远。
- **归因（两个单变量对照，各只改一个参数）**：

  | 配置 | prefill@512 | prefill@1024 | 归因 |
  |---|---|---|---|
  | `int8` len1024 / b4 / KV4096（原） | 157.66 | 312.42 | 基线 |
  | `int8_ctl_kv8k` len1024 / b4 / **KV8192** | 156.38 | 306.64 | **KV 容量：无影响** |
  | `int8_ctl_b1` len1024 / **b1** / KV4096 | 167.12 | 333.33 | **maxBatchSize 4→1：+6.0% / +6.7%** |
  | `int8_len4096` **len4096** / b1 / KV8192 | 181.78 | 360.86 | 再叠加 **opt 位移：+8.8% / +8.3%** |

  两个原因独立且近似可加。**maxBatchSize 4→1 让 batch-1 自己变慢**这点最反直觉——profile 的
  opt 是 `(maxBatchSize, maxInputLen/2)`，batch=4 的 opt 似乎让 TRT 选到了对 batch=1 也更友好的 kernel。
- **为什么只有 INT8**：INT4 是 W4A16（权重反量化后走 fp16 GEMM，tactic 对形状不敏感）；
  INT8 是 W8A8，跑原生 int8 张量核，tactic/tile 空间更大也更挑形状。**机制是推断，未证明。**
- **处置**：老引擎（`engines/int4`、`engines/int8`）**一律保留不动**，所有已发布数字仍出自它们；
  容量升级版另存 `engines/{int4,int8}_len4096`。INT4 的容量升级**零代价**，INT8 要付 ~15%。
- **教训**：
  1. **"把上限调大点"不是免费的**——工具可能把调优点绑在上限上。改容量参数后**必须在原来的工作点上重测**。
  2. 一次改 3 个参数 = 无法归因。**两个单变量对照（共 8 分钟）就把 15% 拆成了 6% + 9% + 0%**，
     还顺手推翻了我自己"batch 4 会 OOM 所以必须降到 1"的猜测：实测 INT8 build 峰值在
     27.6–29.7 GB 之间、与这几个参数几乎无关（由权重 workspace 主导，不是激活 profile），
     len1024→len4096 只多 0.6 GB。**猜的资源账要用实测推翻。**

## H. 口径与约束的事后复核（2026-07-26）

### H1. n=210 基准最长请求距引擎输入上限只剩 9 个 token
> **2026-07-27 追加的更准确说法**：`maxInputLen = 1024` **是 `llm_build` 的默认值**（`--maxInputLen  Default = 1024`），
> 我们从来没有设过它。所以这不是"侥幸通过"，而是**"没做这个选择"**——上限由工具默认值决定，恰好没被撞破。
> 同批发现 `maxBatchSize` 默认 4（我们全程只跑 batch=1）、`maxKVCacheCapacity` 默认 4096。
> 已按实测重建 4096 版引擎，见 §H2。

- **现象**：为回答"我们到底喂了多大的图、多少 token"而复核时发现，基准里 **40 段视频是 5:4**（帧 448×358 → processor 缩到 448×352），
  **154 image token/帧**，而不是此前一律宣称的 112。实测最长的一题（`request_idx 94`）`Computed Tokens = 1015`，
  引擎 `maxInputLen = 1024` ⇒ **余量 9 个 token（99.1% 占满）**。
- **根因**：`Qwen2VLImageProcessor` 把边长缩到 `patch_size × merge_size = 32` 的整数倍，token 数 = (W/32)×(H/32)；
  "112/帧"只对 16:9 的 448×252 成立。我们此前只用 `robovqa_0`（16:9）做视觉扫描，就把结论推广到了全体。
- **影响评估**：基准日志无截断/超限告警、210 条全部返回、正确率可由 `score_mc.py` 复现 ⇒ **已发布结果不受影响**；
  但这是"侥幸通过"，不是"设计通过"。
- **另一处同向约束**：视觉引擎自身 `max_image_tokens = 1024`（`max_image_tokens_per_image = 512`）——
  6×154 = 924 也逼近它。两个 1024 上限（视觉引擎 token 数 / LLM 输入长度）几乎同时到顶。
- **解决**：文档与图表改为按帧形状分别陈述（图 2 给出预处理链与 prompt 预算条形图）；
  要加帧数或加长题干必须以更大 `maxInputLen` 重建引擎。
- **教训**：**"每帧 N token" 不是模型常数，而是 (帧形状, patch, merge) 的函数**。
  凡是"每 X 多少 Y"的结论，都要用**全量数据的实测分布**验证，而不是拿第一个样本外推；
  并且要把**最坏情况**与硬上限一起算出来（我们直到复核才知道余量只有 0.9%）。

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

---
> 说明：**量化校准集用的是通用新闻文本（非驾驶域）** 属于**方法局限/口径**而非 bug，其影响分析见 [METHODOLOGY.md](METHODOLOGY.md) §4 与主页面「思考讨论」。

# 实验方法与可复现性

> 本文件回答四个根本问题:**模型/权重从哪来、数据集是什么、指标怎么定义与测、量化用了什么校准集及其影响**。
> 所有数字均为实测、可复现;无标注/未测的地方明确标注,不外推。更新:2026-07-22。

## 0. 来源与版本总表（全部实测，可追溯）

> 值均取自实际环境（`git rev-parse` / `dpkg -l` / `cat /etc/nv_tegra_release` / `nvcc` / config.json / 运行日志），非记忆、非编造。**未 pin 的版本如实标注**——诚实优先于好看。

| 类别 | 型号 / 版本 | 下载地址 / 取证 | 备注 |
|---|---|---|---|
| **模型** | `nvidia/Cosmos-Reason2-8B` | https://huggingface.co/nvidia/Cosmos-Reason2-8B （**门控**） | NVIDIA 预训练；**我们只推理量化、不训练**。**revision 未 pin**（导出时下载 main 最新，未记 commit）→ 建议 `from_pretrained(revision=…)` 固定 |
| 模型架构（实测） | 36 层 · hidden 4096 · 32 Q 头 · **8 KV 头(GQA)** · headDim 128 · RoPE 262144 · vocab 151936 · eos `[151645,151643]` | `edge_int4/…/llm/config.json` + Orin `LLMEngineConfig` 日志 | Qwen2.5-VL 系 8B backbone；`AutoModelForImageTextToText` 可加载 |
| **量化校准集** | `cnn_dailymail` config **3.0.0**，split `train` 前 **512** 篇 `article` | https://huggingface.co/datasets/cnn_dailymail | 实测 `quantize.py`；纯文本路径，非驾驶域（见 §4） |
| **精度探针集** | 12 条手写 prompt（无标注） | `eval/accuracy/prompts_greedy.json`（本仓库） | 非标准 benchmark |
| **量化/部署框架** | NVIDIA **TensorRT-Edge-LLM v0.9.0** @ `1ac0f2b9…`（2026-07-02） | https://github.com/NVIDIA/TensorRT-Edge-LLM | Orin 建引擎/运行即此 commit（实测 `git rev-parse`）。**Kaggle 量化/导出用 `git clone --depth 1`（main@运行日，未 pin）** → 建议同 pin v0.9.0 |
| 量化后端 | NVIDIA TensorRT-Model-Optimizer (ModelOpt)，随 TRTEdge `.[tools]` 装 | 同上（依赖） | 具体版本随 TRTEdge 依赖，未单独 pin |
| **边缘设备** | **NVIDIA Jetson AGX Orin Developer Kit**（sm_87 Ampere，统一内存 29 GB，**MAXN**） | 实测 `/proc/device-tree/model`、`nvpmodel -q` | — |
| Orin 系统栈 | L4T **R36.4.4**（GCID 41062509，2025-06-16）= JetPack 6.2｜CUDA **12.6.11**｜TensorRT **10.7.0.23**（+cuda12.6）｜Python 3.10.12 | `/etc/nv_tegra_release`、`/usr/local/cuda/version.json`、`dpkg -l` | — |
| **云端**（量化/导出/FP16 参考） | Kaggle：**2× Tesla T4**（15360 MiB，sm_75）｜Python 3.12｜`machine_shape=NvidiaTeslaT4` | kaggle.com（私有 kernel） | torch 预装（支持 sm_70–sm_120，约 2.7，**未 pin**）；transformers 用 `pip install -U 'transformers>=4.51'`（**具体版本未 pin**）→ 建议锁版 |

## 1. 模型与权重来源

| 项 | 事实 | 来源/证据 |
|---|---|---|
| 模型 | `nvidia/Cosmos-Reason2-8B` | HuggingFace(**门控**仓库,需接受许可+授权 token)：https://huggingface.co/nvidia/Cosmos-Reason2-8B |
| 权重 | NVIDIA 预训练权重,**我们未做任何训练/微调** | 直接从 HF 下载 fp16 权重（revision 未 pin，见 §0） |
| 架构(实测 config.json) | 36 decoder 层 · hidden 4096 · 32 注意力头 · **8 KV 头(GQA)** · headDim 128 · RoPE 上下文 262144 · vocab 151936 · eos `[151645,151643]` | `edge_int4/…/llm/config.json` 与 Orin `LLMEngineConfig` 日志一致 |
| 类型 | 物理场景推理**多模态 VLM**(Qwen2.5-VL 系 backbone),HF `AutoModelForImageTextToText` 可加载 | 加载日志 |

我们的工作是**推理侧的量化与边缘部署**,不涉及模型训练。因此"训练集"不适用——预训练数据是 NVIDIA 的,不在本项目范围。

## 2. 数据集(三处,用途不同,都不是我们训练的)

1. **量化校准集(PTQ calibration)** —— 见 §4,`cnn_dailymail` 512 篇。
2. **精度评测基准(主)** —— **官方 [`nvidia/Cosmos-Reason1-Benchmark`](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark) 的 `robovqa` 子集**:110 道**多选题带标准答案**(具身机器人推理,视频+问题)。视频按 **6 帧(≤448px)** 采样作多图,FP16/INT8/INT4 用**同一批帧**→算 **MC 任务正确率**及掉点。产物 `eval/accuracy/benchmark/`。**注**:仅 robovqa 子集、6帧@448 非官方原生视频协议,故绝对分不追求复现论文,但三方同帧→掉点严格可比。
   - **精度探针集(辅)** —— 12 条手写驾驶/推理 prompt(`eval/accuracy/prompts_greedy.json`,无标注),仅用于更敏感的 perplexity 相对指标。
3. **性能基准的输入** —— `llm_bench` 用**合成 token 序列**(指定 `inputLen`/`pastKVLen`),测的是纯 prefill/decode 算力,与内容无关,是延迟基准的标准做法。

> 诚实边界:**没有带标注的下游任务正确率**(无公开 AV-QA 标注集可用+算力/时间约束)。精度结论基于 perplexity/一致率的相对退化,不是任务准确率。

## 3. 指标定义与测量方法(全部可复现)

**环境**（精确版本见 §0）:Jetson AGX Orin Developer Kit (sm_87) / L4T R36.4.4 (JetPack 6.2) / CUDA 12.6.11 / TensorRT 10.7.0.23 / TensorRT-Edge-LLM v0.9.0@`1ac0f2b` / **MAXN**;batch=1。

### 3.1 延迟 / 吞吐 —— `llm_bench`
```bash
# prefill(读题):inputLen=512
llm_bench --engineDir engines/int4/llm --mode prefill --inputLen 512 --iterations 10 --warmup 3
# decode(逐 token 生成):pastKVLen=512,OSL=1(单步),decode 用 CUDA graph
llm_bench --engineDir engines/int4/llm --mode decode  --pastKVLen 512 --iterations 10 --warmup 3
```
- 报告 **E2E Time = mean ± std(10 次迭代,前 3 次 warmup 丢弃)**;`tok/s = 处理 token 数 / E2E`。
- **Prefill**:一次并行处理 512 输入 token(计算密集)。**Decode**:在 512 长度 KV 缓存上生成 1 个 token 的单步延迟(访存密集)。
- 引擎构建参数(三版一致,来自 `LLMEngineConfig` 日志):`maxBatch=4 maxInputLen=1024 maxKVCapacity=4096`。
- **可复现性验证(2026-07-22 重跑 INT4)**:prefill `301.22 ± 0.26 ms / 1699.7 tok·s`、decode `33.93 ± 6.89 ms / 29.5 tok·s`,与首测(300.98 ms/1701、32.34 ms/30.9)在 run-to-run 噪声内一致(decode 单步方差天然大)。
- **标准服务指标映射**:**TTFT**(首 token 延迟)= prefill E2E(随输入长度变化,INT4 128/512/1024-token ≈ 86/301/595 ms);**TPOT/ITL** = decode 单 token 延迟;**输出 TPS** = 1000/TPOT;**E2E**(512-in+128-out)= TTFT + 128·TPOT(INT4 ≈ 4.44 s、INT8 ≈ 6.64 s)。
- **FP16 无设备端基线**:FP16 引擎(~16GB)在 Orin `llm_build` 期 **OOM 被杀(exit 137)**,16GB+TRT builder workspace > 29GB 统一内存 → FP16 的 TTFT/TPS 无法在设备上测得;这本身是结论(量化是部署前提),加速比以显存可行性替代。

### 3.2 功耗 / 能效 —— `tegrastats`
- 在 bench 运行期采样 `tegrastats`,日志留档 `pw_int4_pre.log`/`pw_int4_dec.log`/`pw_int8_*`。
- **整机功耗 = VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0** 三轨之和(mW),取 GPU-active 段均值。
- **能效 tok/J = 吞吐(tok/s) ÷ 平均功耗(W)**。

### 3.3 量化掉点(`eval/accuracy/`)
**主指标 — MC 任务正确率(真实基准)**:在 Cosmos-Reason1-Benchmark(robovqa,110 MC)上,FP16(云端 PyTorch)/INT8/INT4(Orin) 用同一批 6 帧@448 多图,模型贪心输出→正则解析首个 A–D 字母→比标准答案。**结果:FP16 88.2% / INT8 87.3% / INT4 87.3%(−0.9 pts)**,与 FP16 一致率 91.8%。**统计检验:Wilson 95% CI ≈ ±6pts(FP16[80.8,93.0]、INT4/INT8[79.8,92.3]),McNemar p=1.0 → 差异统计不显著**(n=110、二选一,CI 偏宽)。稳妥结论:**量化无显著任务正确率损失**;要分辨 INT4/INT8 需扩样本+更难多选题。产物 `eval/accuracy/benchmark/RESULTS.json`。

**辅助指标 — perplexity + token 一致率(12 prompt 探针)**:
- FP16 参考在 **Kaggle T4×2** 用 PyTorch 加载原模型(Orin 无外网、8B fp16≈16GB 本地装不下)。
- **perplexity(teacher-forced)**:对每段输出,拼接 `formatted_prompt + answer`,掩掉 prompt token(label=-100),用 **FP16 模型**前向得答案 token 的平均 NLL,再 `exp` → perplexity。"原模型对该输出有多惊讶",越低越贴近 FP16 分布。相对增幅(vs fp16 自身输出)是退化信号。
- **token 一致前缀**:同一 prompt 贪心(`top_k=1` 确定性),量化输出与 FP16 输出逐 token 相同的最长开头长度 / 占比。
- **口径**:对比的是**已部署 Orin TRT 引擎输出 vs FP16 PyTorch**,故为"量化+引擎实现+后端"的**总部署差距**,非纯量化误差;绝对 PPL 偏低(高概率贪心序列),**看相对增幅**。

## 4. 量化配置与校准集(关键,含影响分析)

**两版均为 PTQ(训练后量化),经 NVIDIA ModelOpt,均需校准集。我们用的是工具默认值**(kaggle 导出脚本 `cloud/kaggle_cosmos_only.py` 调用 `tensorrt-edgellm-quantize llm` 时**未覆盖** `--dataset`/`--num_samples`)。

| 项 | 事实(实测 quantize.py) |
|---|---|
| 校准集 | **`cnn_dailymail` config 3.0.0,train split 前 512 篇 `article`**(通用英文新闻文本；https://huggingface.co/datasets/cnn_dailymail ) |
| 校准样本数 | **512**(`--num_samples` 默认) |
| 校准路径 | **纯文本** `_text_calib_dataloader`(不走视觉塔;calib batch:AWQ=16 / SmoothQuant=1) |
| INT4 (AWQ, W4A16) | 权重 4bit group-wise + 激活 16bit;用校准算**激活感知 per-channel pre-quant scales**,保护显著权重通道 |
| INT8 (SmoothQuant, W8A8) | 权重 8bit + 激活 8bit;用校准的**激活统计**算 smoothing scales,把激活离群值迁移进权重 |
| 视觉塔 | **未量化**(工具仅支持 fp8/fp16,两版都留 fp16)→ 多模态视觉路径不受 LLM 量化影响 |

### 影响分析（已做消融，见 `eval/accuracy/ablation_calibration/`）
- **原假设**:新闻域校准与部署域错配 → 拖累 INT8(W8A8 连激活也量化,对校准更敏感)。
- **消融**:用 Cosmos-Reason1-Benchmark 非评测子集问题作**域内校准集**(400问→100条拼接)重量化 INT8,对比默认 news-512。评测集不重叠、无泄漏。
- **结果:假设被推翻**。域内校准反而**大幅劣化** INT8:perplexity **+244%**(vs news 的 +18.1%)、robovqa MC 84.5%(vs 87.3%),生成还出现错误物理。
- **更深结论**:**校准集的覆盖度/质量(数量、长度、多样性)远比"域匹配"重要**——512 篇长新闻是好的通用校准集,100 条短问题覆盖差→SmoothQuant 激活 scale 估计差→严重劣化。**故默认 news-512 是合理选择,得到验证**;INT8 对校准的敏感性确认,但解药是"好覆盖"而非"域内"。
- **Caveat**:本消融"域"与"规模/长度"混杂(同时变),不能纯归因于域;要纯隔离需同规模/长度的域内 vs 通用集。

## 5. 一键复现指引

- 量化+导出(云,一次性):`cloud/kaggle_cosmos_only.py`(INT4/INT8)、`cloud/kaggle_fp16_ref.py`(FP16 精度参考)。默认校准集即 cnn_dailymail/512;要换域改 `--dataset`/`--num_samples`。
- Orin 建引擎:`llm_build --onnxDir <onnx/llm> --engineDir <out>`(精度由 ONNX 决定)。
- 性能:§3.1 的 `llm_bench` 命令;功耗:并行 `tegrastats`。
- 掉点:`eval/accuracy/`(prompts + Orin 输出 + FP16 参考 + `compute_dropoff.py` + README)。

> 一句话:**模型是 NVIDIA 预训练的 Cosmos-Reason2-8B(我们只做推理量化);校准集是默认的 512 篇 CNN/DailyMail 新闻文本(非驾驶域,这是重要 caveat);精度是小规模手写探针上的相对退化(非标注任务正确率);性能/功耗经 llm_bench+tegrastats 实测且已复现。**

# 实验方法与可复现性

> 本文件回答四个根本问题:**模型/权重从哪来、数据集是什么、指标怎么定义与测、量化用了什么校准集及其影响**。
> 所有数字均为实测、可复现;无标注/未测的地方明确标注,不外推。更新:2026-07-22。

## 1. 模型与权重来源

| 项 | 事实 | 来源/证据 |
|---|---|---|
| 模型 | `nvidia/Cosmos-Reason2-8B` | HuggingFace(**门控**仓库,需接受许可+授权 token) |
| 权重 | NVIDIA 预训练权重,**我们未做任何训练/微调** | 直接从 HF 下载 fp16 权重 |
| 架构(实测 config.json) | 36 decoder 层 · hidden 4096 · 32 注意力头 · **8 KV 头(GQA)** · headDim 128 · RoPE 上下文 262144 · vocab 151936 · eos `[151645,151643]` | `edge_int4/…/llm/config.json` 与 Orin `LLMEngineConfig` 日志一致 |
| 类型 | 物理场景推理**多模态 VLM**(Qwen2.5-VL 系 backbone),HF `AutoModelForImageTextToText` 可加载 | 加载日志 |

我们的工作是**推理侧的量化与边缘部署**,不涉及模型训练。因此"训练集"不适用——预训练数据是 NVIDIA 的,不在本项目范围。

## 2. 数据集(三处,用途不同,都不是我们训练的)

1. **量化校准集(PTQ calibration)** —— 见 §4,`cnn_dailymail` 512 篇。
2. **精度评测探针集** —— **12 条我们手写的**驾驶物理/决策/常识 prompt(`eval/accuracy/prompts_greedy.json`)。**无 ground-truth 标注**;perplexity/一致率指标不需要标注。**这不是标准 benchmark**(未跑 MMLU/MMMU 等),是小规模定性探针,结论仅在此集上成立。
3. **性能基准的输入** —— `llm_bench` 用**合成 token 序列**(指定 `inputLen`/`pastKVLen`),测的是纯 prefill/decode 算力,与内容无关,是延迟基准的标准做法。

> 诚实边界:**没有带标注的下游任务正确率**(无公开 AV-QA 标注集可用+算力/时间约束)。精度结论基于 perplexity/一致率的相对退化,不是任务准确率。

## 3. 指标定义与测量方法(全部可复现)

**环境**:Jetson Orin (sm_87) / JetPack 6 / CUDA 12.6 / TRT 10.7 / **MAXN** 功耗模式;batch=1。

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

### 3.2 功耗 / 能效 —— `tegrastats`
- 在 bench 运行期采样 `tegrastats`,日志留档 `pw_int4_pre.log`/`pw_int4_dec.log`/`pw_int8_*`。
- **整机功耗 = VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0** 三轨之和(mW),取 GPU-active 段均值。
- **能效 tok/J = 吞吐(tok/s) ÷ 平均功耗(W)**。

### 3.3 量化掉点 —— FP16 参考(`eval/accuracy/`)
- FP16 参考在 **Kaggle T4×2** 用 PyTorch 加载原模型(Orin 无外网、8B fp16≈16GB 本地装不下)。
- **perplexity(teacher-forced)**:对每段输出,拼接 `formatted_prompt + answer`,掩掉 prompt token(label=-100),用 **FP16 模型**前向得答案 token 的平均 NLL,再 `exp` → perplexity。"原模型对该输出有多惊讶",越低越贴近 FP16 分布。相对增幅(vs fp16 自身输出)是退化信号。
- **token 一致前缀**:同一 prompt 贪心(`top_k=1` 确定性),量化输出与 FP16 输出逐 token 相同的最长开头长度 / 占比。
- **口径**:对比的是**已部署 Orin TRT 引擎输出 vs FP16 PyTorch**,故为"量化+引擎实现+后端"的**总部署差距**,非纯量化误差;绝对 PPL 偏低(高概率贪心序列),**看相对增幅**。

## 4. 量化配置与校准集(关键,含影响分析)

**两版均为 PTQ(训练后量化),经 NVIDIA ModelOpt,均需校准集。我们用的是工具默认值**(kaggle 导出脚本 `cloud/kaggle_cosmos_only.py` 调用 `tensorrt-edgellm-quantize llm` 时**未覆盖** `--dataset`/`--num_samples`)。

| 项 | 事实(实测 quantize.py) |
|---|---|
| 校准集 | **`cnn_dailymail` 3.0.0,train split 前 512 篇 `article`**(通用英文新闻文本) |
| 校准样本数 | **512**(`--num_samples` 默认) |
| 校准路径 | **纯文本** `_text_calib_dataloader`(不走视觉塔;calib batch:AWQ=16 / SmoothQuant=1) |
| INT4 (AWQ, W4A16) | 权重 4bit group-wise + 激活 16bit;用校准算**激活感知 per-channel pre-quant scales**,保护显著权重通道 |
| INT8 (SmoothQuant, W8A8) | 权重 8bit + 激活 8bit;用校准的**激活统计**算 smoothing scales,把激活离群值迁移进权重 |
| 视觉塔 | **未量化**(工具仅支持 fp8/fp16,两版都留 fp16)→ 多模态视觉路径不受 LLM 量化影响 |

### 影响分析(机制明确;因果为"合理推断",未做消融)
- **域错配**:校准用**新闻文本**,部署是**驾驶场景推理**。校准分布 ≠ 部署分布,可能低估驾驶 prompt 关键通道的激活范围,放大量化误差。
- **为何 INT8 掉点反而更大(+18.1% vs INT4 +9.8%)**:SmoothQuant **连激活也量化(W8A8)**,对校准质量/域错配更敏感;AWQ 是**纯权重(W4A16)**,激活保持 16bit,对域错配更鲁棒。这与观测方向一致,**是合理主因之一,但未经消融证明**。
- **可验证的下一步(未做)**:用**驾驶域校准集**(如 PhysicalAI-AV 文本/多模态)重量化,对比掉点,即可确认域错配的贡献。这是明确的后续实验,不在当前结论内。

## 5. 一键复现指引

- 量化+导出(云,一次性):`cloud/kaggle_cosmos_only.py`(INT4/INT8)、`cloud/kaggle_fp16_ref.py`(FP16 精度参考)。默认校准集即 cnn_dailymail/512;要换域改 `--dataset`/`--num_samples`。
- Orin 建引擎:`llm_build --onnxDir <onnx/llm> --engineDir <out>`(精度由 ONNX 决定)。
- 性能:§3.1 的 `llm_bench` 命令;功耗:并行 `tegrastats`。
- 掉点:`eval/accuracy/`(prompts + Orin 输出 + FP16 参考 + `compute_dropoff.py` + README)。

> 一句话:**模型是 NVIDIA 预训练的 Cosmos-Reason2-8B(我们只做推理量化);校准集是默认的 512 篇 CNN/DailyMail 新闻文本(非驾驶域,这是重要 caveat);精度是小规模手写探针上的相对退化(非标注任务正确率);性能/功耗经 llm_bench+tegrastats 实测且已复现。**

# 量化掉点评测（vs FP16 基线）

补齐"精度轴"：量化 INT4/INT8 相对 FP16 原模型掉多少。结果与结论见
[`../../docs/edge_deploy_status.md`](../../docs/edge_deploy_status.md) 的"量化掉点"节。

## 为什么这样测

- Orin **无外网**、8B fp16 ≈16GB 本地 3070(8GB) 装不下 → FP16 参考只能在**云端一次性**产出（Kaggle T4×2 PyTorch）。
- 生成式 VLM 无干净的单一"正确率"；且无带标注的 AV QA 集。故用两个可复现的相对指标：
  1. **teacher-forced perplexity**：用 FP16 模型对 INT4/INT8 的**实际输出文本**打分（掩掉 prompt、只算答案 token 的平均 NLL → exp）。"原模型对量化输出有多惊讶"，越低=越贴近 FP16 分布。
  2. **token 一致前缀**：INT4/INT8 贪心输出与 FP16 贪心输出的最长公共 token 前缀。

## 口径与边界（诚实）

- PPL 是对各自**贪心输出**（高概率序列）的打分，绝对值偏低（~1.2–1.5）；**有意义的是相对增幅**。
- 对比的是**已部署的 Orin TRT 引擎**输出 vs **FP16 PyTorch**，故捕获"量化+引擎实现+后端"的**总部署差距**（实用相关：相比原模型你实际损失多少），非纯量化误差。token 一致前缀偏低正是贪心下后端差异逐 token 累积——因此 perplexity 为主、一致率为辅。
- FP16 参考在 T4 上以 fp16 加载（bf16/fp16 差异对退化参考可忽略），与 Orin fp16 派生引擎口径一致。

## 结果（12 条驾驶/推理 prompt，贪心 top_k=1）

| 指标 | FP16 | INT4 (AWQ) | INT8 (SmoothQuant) |
|---|---|---|---|
| 平均 perplexity | 1.237 | 1.359 (+9.8%) | 1.460 (+18.1%) |
| 与 FP16 贪心一致前缀占比 | 100% | 16.4% | 5.7% |

**INT4(AWQ) 掉点小于 INT8(SmoothQuant)**——W4A16 纯权重量化比 W8A8（连激活也量化）更保输出分布。

## 复现

```
# 1) Orin 端：贪心确定性生成（top_k=1）——见 prompts_greedy.json
llm_inference --engineDir <int4|int8 llm> --multimodalEngineDir <int4 visual> \
  --inputFile prompts_greedy.json --outputFile orin_<q>_outputs.json --maxGenerateLength 200
# 2) 云端 FP16 参考 + 打分：把 orin_*_outputs.json 注入 cloud/kaggle_fp16_ref.py 的 CANDIDATES，
#    kaggle kernels push（machine_shape=NvidiaTeslaT4 锁 T4×2；HF_TOKEN 需有 Cosmos 门控权限）→ fp16_reference.json
# 3) 汇总：
python compute_dropoff.py fp16_reference.json
```

## 文件

- `prompts_greedy.json` — 12 条固定 prompt（贪心参数）。
- `orin_int4_outputs.json` / `orin_int8_outputs.json` — Orin 引擎贪心实际输出。
- `fp16_reference.json` — 云端 FP16 生成 + 对三方输出的 PPL/一致率打分（kernel 产物）。
- `compute_dropoff.py` — 汇总成掉点表。
- kernel 源码（去 token）见 [`../../cloud/kaggle_fp16_ref.py`](../../cloud/kaggle_fp16_ref.py)。

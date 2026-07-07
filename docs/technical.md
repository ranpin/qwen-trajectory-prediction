# 技术文档

本文档面向开发者，覆盖架构、数据管线、Prompt 规范、训练、量化、部署、评估方法学。
产品层面的目标与规划见 [PRD.md](../PRD.md)；快速上手见 [README.md](../README.md)。

---

## 1. 架构总览

```
                合成数据生成 ┐
                            ├─► SFT 数据(chat) ─► QLoRA 微调 ─► 合并 ─► GGUF 量化
   ETH/UCY 解析 + 归一化 ────┘        (pc-3070, RTX 3070 8GB)              │
                                                                          ▼
   浏览器 ──HTTP──► Gradio Demo ──HTTP──► llama.cpp server (Jetson Orin) ◄─ 部署
```

轨迹预测被建模为**文本生成**：把观测轨迹坐标写成文本，模型输出未来坐标 + 自然语言解释。

---

## 2. 数据管线

### 2.1 序列化格式（ms-swift chat）

每条样本是一个三段式对话：`system`（角色设定）/ `user`（观测 + 场景 + 速度方向）/
`assistant`（趋势分析 + 预测坐标）。观测 8 帧（3.2s），预测 12 帧（4.8s），帧间隔 0.4s。

### 2.2 合成数据（`scripts/data_prep/synthetic_gen.py`）

四种运动学模型：匀速直线、匀转弯、匀加速、走停(stop-and-go)，叠加高斯噪声。
`--num_samples` 控制数量，`--normalize` 控制坐标归一化（见 2.4）。

### 2.3 真实数据 ETH/UCY（`download_datasets.py` + `preprocess.py`）

- 格式：制表符分隔 `frame_id ped_id x y`，世界坐标（米），标注每 10 帧一次。
- **帧步长探测**：`extract_samples` 会按每个行人的最小相邻帧差探测采样步长
  （ETH/UCY=10，合成=1）。早期版本把步长硬编码为 1，导致真实数据**一条窗口都提取不到**——已修。
- 滑窗要求窗口内帧号严格等间隔、无缺帧。

### 2.4 坐标归一化（`scripts/common/trajectory_norm.py`）

真实世界绝对坐标（如 13.6, 5.8）远离合成训练分布，损害精度。归一化把每条轨迹变换到
**agent-centric 帧**，让模型只需学习相对运动：

| 模式 | 变换 |
|------|------|
| `none` | 恒等 |
| `translate` | 观测末点 → 原点 |
| `translate_rotate` | 观测末点 → 原点，且观测整体朝向旋转到 +x 轴 |

关键性质：变换**只由观测决定**（因果、不泄露未来），对观测与未来施加同一变换，**可逆**。
因此变换完全由观测确定，Demo 在推理时重新计算，无需持久化到训练数据。ADE/FDE 是距离度量，
对平移/旋转**不变**，所以归一化前后指标可直接对比。

---

## 3. Prompt 格式规范（`scripts/common/prompt_format.py`）

训练数据、部署模型、Demo 必须使用**完全一致**的 user prompt。规范构造器
`build_user_prompt()` 是唯一真源，`preprocess.py` / `synthetic_gen.py` / `demo/app.py` 全部 import 它。

> 历史教训：早期 Demo 自行拼 `场景：X\n历史轨迹：Y`，与训练格式不一致，导致现场推理效果差于评估。

user prompt 模板：

```
场景：{scene}
行人历史轨迹（过去3.2秒，每0.4秒采样）：
t=0.0s: (x, y)
... 8 行 ...
平均速度：{v}m/s，主要方向：{dir}

请预测该行人未来4.8秒的轨迹。
```

---

## 4. 训练（QLoRA）

在 pc-3070（RTX 3070 8GB）上用 ms-swift 做 QLoRA 4-bit 微调。核心超参
（详见 `scripts/training/train_synthetic.sh`）：

| 参数 | 值 |
|------|-----|
| lora_rank / alpha | 16 / 32 |
| learning_rate | 1e-4 |
| quant | bnb 4-bit |
| batch / grad_accum | 2 / 8（有效 16）|
| max_length | 1024 |
| max_steps | 500（合成数据）|

训练后 `swift export --merge_lora` 合并权重。

---

## 5. 量化

| 方法 | 工具 | 说明 |
|------|------|------|
| GGUF Q4_K_M | llama.cpp | 端侧首选，CPU+GPU 混合；`scripts/training/quantize_gguf.py` |
| AWQ 4-bit | ms-swift/AutoAWQ | 备选；`scripts/training/quantize_awq.py` |

Q4_K_M 约 2.4GB。

---

## 6. Orin 部署

- llama.cpp `llama-server`，`-ngl 99` 全层 offload，端口 8080，OpenAI 兼容 API。
- 部署脚本 `scripts/deploy/deploy_to_orin.sh <model.gguf> <host>`。
- **端点配置集中在 `configs/deploy.env`**（`ORIN_API_URL` / `ORIN_MODEL` / `ORIN_HOST` /
  `ORIN_SSH_USER`）。评估与 Demo 脚本默认读这些环境变量，不再硬编码 IP。
  SSH 密码通过 `ORIN_SSH_PASSWORD` 环境变量传入，不入库。

```bash
set -a && . configs/deploy.env && set +a
```

---

## 7. 评估方法学

### 7.1 指标定义与口径

- **单次预测 ADE/FDE**：每条轨迹只预测 1 条。**不是**文献常用的 best-of-K
  （minADE_K / minFDE_K），不可与 best-of-20 榜单直接对比。
- ADE = 各时刻位移误差均值；FDE = 终点误差；Miss Rate = FDE>2m 的比例。
- 均值附 **95% 置信区间**（正态近似）。

### 7.2 稳健性（`scripts/evaluation/evaluate.py`）

- **按 prompt 文本对齐**预测与真值（而非文件行位置），预测不全时也不会静默错位。
- 解析只取 `预测轨迹` 标记之后的坐标，避免模型回显历史点污染对齐。
- 报告 `unmatched` / `parse_failures` / `length_mismatches` 计数。

### 7.3 CVM 基线（`scripts/evaluation/baselines.py`）

匀速模型（观测段平均速度线性外推）是 ETH/UCY 单模态 ADE/FDE 上著名的强基线，是判断
“LLM 是否比平凡外推更有价值”的关键对照。基线预测以**与模型完全相同的文本格式**输出，直接复用
`evaluate.py` 打分。

### 7.4 当前结果（单次预测）

| 测试集 | LLM ADE | CVM ADE | LLM FDE | CVM FDE | LLM MR | CVM MR |
|--------|---------|---------|---------|---------|--------|--------|
| 合成 (190/200) | **0.82** | 1.53 | **1.37** | 2.70 | **12.1%** | 48% |
| ETH/UCY (141/147) | 0.79 | **0.65** | 1.62 | **1.38** | 24.1% | **22.7%** |

**结论**：真实数据上 CVM 反超 LLM，LLM 仅在自身合成分布上占优。归一化重训（第 8 节）用于验证能否翻盘。

---

## 8. 归一化重训 A/B（A3）

一键脚本（在 pc-3070 运行）：

```bash
bash scripts/training/retrain_normalized.sh translate_rotate
```

它会：生成归一化合成数据 → QLoRA 训练+合并 → GGUF 量化 → 部署到 Orin，并打印随后的评估命令。
评估用 `preprocess.py --normalize` 造归一化测试集，`sample_testset.py` 抽同一批 150 条，再
`generate_predictions.py` + `evaluate.py` 打分。因 ADE/FDE 平移旋转不变，CVM 列不变，只看 LLM 列是否超过 CVM。

若 `translate_rotate` 不升反降，回退 `translate`；结果表如实记录，不粉饰。

---

## 9. 复现命令汇总

```bash
# 数据
python scripts/data_prep/download_datasets.py
python scripts/data_prep/preprocess.py [--normalize translate_rotate]
python scripts/data_prep/synthetic_gen.py --num_samples 50000 [--normalize translate_rotate]

# 评估（LLM）
set -a && . configs/deploy.env && set +a
python scripts/evaluation/sample_testset.py --test_file data/processed/trajectory_test.jsonl \
    --output_file data/processed/eth_ucy_test_sample.jsonl
python scripts/evaluation/generate_predictions.py --test_file data/processed/eth_ucy_test_sample.jsonl \
    --output_file data/processed/eth_ucy_predictions.jsonl --max_samples 150
python scripts/evaluation/evaluate.py --test_file data/processed/eth_ucy_test_sample.jsonl \
    --predictions_file data/processed/eth_ucy_predictions.jsonl --output_file outputs/eval_results_eth_ucy.json

# 评估（CVM 基线）
python scripts/evaluation/baselines.py --model cvm --test_file data/processed/eth_ucy_test_sample.jsonl \
    --output_file data/processed/cv_predictions_eth_ucy.jsonl
python scripts/evaluation/evaluate.py --test_file data/processed/eth_ucy_test_sample.jsonl \
    --predictions_file data/processed/cv_predictions_eth_ucy.jsonl --output_file outputs/eval_results_cv_eth_ucy.json

# Demo
python demo/app.py --normalize none   # 与部署模型的归一化模式一致
```

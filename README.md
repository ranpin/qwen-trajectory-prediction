# 基于Qwen3-4B的轨迹预测系统

> 微调 → 量化加速 → NVIDIA Orin端侧部署 全链路实现

## 项目简介

本项目基于Qwen3-4B大语言模型，实现一个完整的轨迹预测系统。通过将轨迹预测建模为文本生成任务，利用LLM的语义理解能力进行可解释的轨迹预测，并通过量化加速在NVIDIA Jetson Orin上实现实时推理。

## 核心特性

- **LLM驱动**：基于Qwen3-4B微调，支持自然语言解释预测结果
- **轻量微调**：使用QLoRA 4-bit，单张RTX 3070（8GB）即可完成训练
- **量化加速**：AWQ/GGUF 4-bit量化，模型压缩至2GB
- **端侧部署**：llama.cpp在Jetson Orin上实时推理
- **交互式Demo**：Gradio可视化界面，支持在线演示

## 技术栈

| 环节 | 技术选型 |
|------|----------|
| 基座模型 | Qwen3-4B |
| 微调框架 | ms-swift (QLoRA 4-bit) |
| 数据集 | ETH/UCY（真实评估）+ 合成数据（训练，50000 样本）|
| 量化 | AWQ 4-bit / GGUF Q4_K_M |
| 推理引擎 | llama.cpp |
| 部署目标 | NVIDIA Jetson AGX Orin 64GB |
| Demo | Gradio Blocks |

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+
- RTX 3070 (8GB) 或更高

### 安装依赖

```bash
git clone https://github.com/ranpin/qwen-trajectory-prediction.git
cd qwen-trajectory-prediction
pip install -r requirements.txt
```

> 完整参数、设计与方法学见 [docs/technical.md](docs/technical.md)。

### 数据准备

```bash
python scripts/data_prep/download_datasets.py                       # 下载 ETH/UCY
python scripts/data_prep/preprocess.py                             # 真实数据 -> chat 格式
python scripts/data_prep/synthetic_gen.py --num_samples 50000      # 生成合成训练数据
# 可加 --normalize translate_rotate 生成 agent-centric 归一化数据
```

### 微调 → 量化 → 部署（在 pc-3070 上）

```bash
bash scripts/training/train_synthetic.sh                           # QLoRA 训练 + 合并
python scripts/training/quantize_gguf.py \
    --model_path outputs/qwen3-4b-synthetic-lora-merged --quant_type Q4_K_M
set -a && . configs/deploy.env && set +a                          # 配置 Orin 端点
bash scripts/deploy/deploy_to_orin.sh outputs/qwen3-4b-q4_k_m.gguf "$ORIN_HOST"
# 归一化重训一键脚本：bash scripts/training/retrain_normalized.sh translate_rotate
```

### 运行 Demo

```bash
set -a && . configs/deploy.env && set +a
python demo/app.py --normalize none      # --normalize 需与部署模型一致
```

## 项目结构

```
qwen-trajectory-prediction/
├── data/
│   ├── raw/              # 原始数据集
│   ├── processed/        # 预处理后的数据
│   └── synthetic/        # 合成数据
├── configs/              # 配置文件
├── scripts/
│   ├── common/           # 共享模块（prompt 格式、坐标归一化）
│   ├── data_prep/        # 数据准备脚本
│   ├── training/         # 训练和量化脚本
│   ├── evaluation/       # 评估、基线、抽样脚本
│   └── deploy/           # 部署脚本
├── notebooks/            # Jupyter notebooks
├── demo/                 # Gradio Demo
├── docs/                 # 文档（PRD.md、technical.md）
├── outputs/              # 模型输出（gitignore）
└── README.md
```

## 评估指标

目标值：

| 指标 | 目标值 |
|------|--------|
| ADE | <0.5m |
| FDE | <1.0m |
| Miss Rate | <20% |
| 推理延迟 | <300ms (Orin) |

### 实际评估结果

> 说明：下表为**单次预测（single-prediction）** 的 ADE/FDE，**不是**文献常用的
> best-of-K（minADE_K / minFDE_K）指标，不可直接与 best-of-20 榜单对比。
> 误差均值附 95% 置信区间；评估脚本按 prompt 文本对齐预测与真值。

**LLM vs CVM 匀速基线**（同口径）：

| 测试集 | LLM ADE | CVM ADE | LLM FDE | CVM FDE | LLM MR | CVM MR |
|--------|---------|---------|---------|---------|--------|--------|
| 合成 (190/200) | **0.82 ± 0.29** | 1.53 | **1.37 ± 0.39** | 2.70 | **12.1%** | 48% |
| ETH/UCY (141/147) | 0.79 ± 0.17 | **0.65** | 1.62 ± 0.35 | **1.38** | 24.1% | **22.7%** |
| ETH/UCY + 归一化重训 | 🔄 待测 | 0.65 | 🔄 待测 | 1.38 | 🔄 | 22.7% |

> ⚠️ **关键发现**：真实数据上匀速基线(CVM)反超 LLM；LLM 仅在自身合成分布上占优。
> 归一化重训（A3）用于验证能否翻盘。方法学、复现命令见
> [docs/technical.md](docs/technical.md) 第 7–9 节。指标为**单次预测**，非 best-of-K。

## 文档

- [docs/PRD.md](docs/PRD.md) — 产品需求（目标、里程碑、风险、方向）
- [docs/technical.md](docs/technical.md) — 技术细节（架构、数据、训练、量化、部署、评估方法学）

## 开发计划

- [x] 项目调研与规划
- [x] GitHub仓库创建
- [x] 数据准备与预处理（合成 + ETH/UCY）
- [x] 模型微调（QLoRA）
- [x] 量化加速（GGUF Q4_K_M）
- [x] Orin部署（llama.cpp 服务）
- [x] Demo开发（Gradio）
- [x] 模型评估（合成 + 真实基准 + CVM 基线）
- [x] 文档完善（PRD / README / technical 三拆）
- [ ] 坐标归一化重训 A/B（待 pc-3070 执行）

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

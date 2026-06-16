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
| 数据集 | ETH/UCY + TrajNet++ + 合成数据 |
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

### 数据准备

```bash
# 下载数据集
bash scripts/data_prep/download_datasets.sh

# 生成合成数据
python scripts/data_prep/synthetic_gen.py --num_samples 100000

# 序列化为训练格式
python scripts/data_prep/preprocess.py --output data/processed/trajectory_sft.json
```

### 模型微调

```bash
# QLoRA微调
swift sft \
    --model Qwen/Qwen3-4B \
    --tuner_type lora \
    --dataset data/processed/trajectory_sft.json \
    --output_dir outputs/qwen3-4b-trajectory-lora \
    --num_train_epochs 3
```

### 量化与部署

```bash
# AWQ量化
python scripts/training/quantize_awq.py

# GGUF量化（用于llama.cpp）
python scripts/training/quantize_gguf.py

# 部署到Orin
bash scripts/deploy/deploy_to_orin.sh
```

### 运行Demo

```bash
python demo/app.py --server 0.0.0.0 --port 7860
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
│   ├── data_prep/        # 数据准备脚本
│   ├── training/         # 训练和量化脚本
│   ├── evaluation/       # 评估脚本
│   └── deploy/           # 部署脚本
├── notebooks/            # Jupyter notebooks
├── demo/                 # Gradio Demo
├── docs/                 # 文档
├── outputs/              # 模型输出（gitignore）
├── PRD.md                # 产品需求文档
└── README.md
```

## 评估指标

| 指标 | 目标值 |
|------|--------|
| minADE | <0.5m |
| minFDE | <1.0m |
| Miss Rate | <20% |
| 推理延迟 | <300ms (Orin) |

## 文档

- [PRD.md](PRD.md) - 完整产品需求文档
- [docs/technical.md](docs/technical.md) - 技术架构文档
- [docs/demo_guide.md](docs/demo_guide.md) - Demo使用指南

## 开发计划

- [x] 项目调研与规划
- [x] GitHub仓库创建
- [ ] 数据准备与预处理
- [ ] 模型微调
- [ ] 量化加速
- [ ] Orin部署
- [ ] Demo开发
- [ ] 文档完善

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

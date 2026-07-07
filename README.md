# 基于 Qwen3-4B 的车辆轨迹预测系统

> 微调 → 量化加速 → NVIDIA Orin 端侧部署 全链路实现（车辆为主，兼容行人）

## 项目简介

本项目基于 Qwen3-4B 大语言模型，实现一个完整的**车辆轨迹预测**系统。将轨迹预测建模为文本生成任务，利用 LLM 的语义理解能力进行可解释的轨迹预测，并通过量化加速在 NVIDIA Jetson Orin 上实现端侧推理。通过 **agent 类型**抽象，同一套流程/模型同时支持车辆（主）与行人。

## 核心特性

- **车辆为主、多 agent 类型**：车辆/行人统一建模，prompt 内以 agent 标签区分
- **LLM 驱动**：基于 Qwen3-4B 微调，支持自然语言解释预测结果
- **轻量微调**：QLoRA 4-bit，单张 RTX 3070（8GB）即可完成训练
- **量化加速**：AWQ/GGUF 4-bit 量化，模型压缩至约 2.4GB
- **端侧部署**：llama.cpp 在 Jetson Orin 上推理
- **交互式 Demo**：Gradio 可视化，含 CVM 匀速基线对比

## 技术栈

| 环节 | 技术选型 |
|------|----------|
| 基座模型 | Qwen3-4B |
| 微调框架 | ms-swift (QLoRA 4-bit) |
| 数据集 | 合成车辆（自行车模型，训练）+ NGSIM（真实车辆评估）+ ETH/UCY（行人）|
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
# 合成车辆数据（自行车模型；mixed=车辆为主、含行人）
python scripts/data_prep/synthetic_gen.py --agent_type mixed --num_samples 50000 \
    --normalize translate_rotate
# 真实车辆（NGSIM，需手动下载 CSV 后解析；见 technical.md 数据获取）
python scripts/data_prep/preprocess_ngsim.py --csv <ngsim.csv>
# 行人（ETH/UCY，可选）
python scripts/data_prep/download_datasets.py && python scripts/data_prep/preprocess.py
```

### 微调 → 量化 → 部署（在 pc-3070 上）

```bash
# 车辆一键重训（生成数据 → QLoRA → 合并 → GGUF）
bash scripts/training/retrain_vehicle.sh translate_rotate
set -a && . configs/deploy.env && set +a                          # 配置 Orin 端点
bash scripts/deploy/deploy_to_orin.sh outputs/qwen3-4b-vehicle-q4_k_m.gguf "$ORIN_HOST"
# 行人归一化重训：bash scripts/training/retrain_normalized.sh translate_rotate
```

### 运行 Demo

```bash
set -a && . configs/deploy.env && set +a
python demo/app.py --agent_type vehicle --normalize translate_rotate
# --agent_type / --normalize 需与所部署模型一致
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

**LLM vs CVM 匀速基线**（同口径，ADE/FDE 单位米）：

车辆（主）：

| 测试集 | LLM ADE | CVM ADE | LLM FDE | CVM FDE | LLM MR | CVM MR |
|--------|---------|---------|---------|---------|--------|--------|
| 合成车辆 (148/200, translate_rotate) | **1.58 ± 0.17** | 7.41 | **3.43 ± 0.40** | 15.29 | 68.2% | 66.5% |
| NGSIM 真实车辆 | ⏳ 待数据 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

行人（兼容）：

| 测试集 | LLM ADE | CVM ADE | LLM FDE | CVM FDE | LLM MR | CVM MR |
|--------|---------|---------|---------|---------|--------|--------|
| 合成 (190/200) | **0.82 ± 0.29** | 1.53 | **1.37 ± 0.39** | 2.70 | **12.1%** | 48% |
| ETH/UCY (141/147) | 0.79 ± 0.17 | **0.65** | 1.62 ± 0.35 | **1.38** | 24.1% | **22.7%** |
| ETH/UCY + 归一化重训 | 0.81 (0.49中位) | **0.65** | 1.73 | **1.37** | 31.6% | **22.7%** |

> ⚠️ **关键发现（行人真实数据）**：匀速基线(CVM)反超 LLM，LLM 仅在自身合成分布上占优。
> **坐标归一化重训未能翻盘**：归一化后 ADE 0.79→0.81、MR 24%→32%（不升反微降），仍明显不及 CVM 0.65。
> 说明 agent-centric 归一化不是症结；纯文本 LLM（不喂地图）在真实行人数据上难敌平凡外推。
> **车辆（合成）**：LLM ADE 1.58 vs CVM 7.41 —— LLM 大胜（学到了转弯/变道/加减速，CVM 只会直线外推）。
> 但这是**同分布**优势；**真实高速数据（NGSIM）才是决定性检验**：高速近似匀速、CVM 极强
> （合成 NGSIM 直行格式上 CVM ADE 仅约 0.16m），纯文本 LLM 不喂地图大概率难赢——该评估仍待数据。
> 指标为**单次预测**（非 best-of-K）；方法学与复现见 [docs/technical.md](docs/technical.md)。
> ⏳ = 待 NGSIM 真实数据（data.transportation.gov 对数据中心 IP 返回 403，需手动下载）。

## 文档

- [docs/PRD.md](docs/PRD.md) — 产品需求（目标、里程碑、风险、方向）
- [docs/technical.md](docs/technical.md) — 技术细节（架构、数据、训练、量化、部署、评估方法学）

## 开发计划

- [x] 项目调研与规划
- [x] GitHub仓库创建
- [x] 数据管线（合成车辆/行人 + NGSIM 解析 + ETH/UCY）
- [x] agent 类型抽象（车辆/行人统一 prompt）
- [x] 合成车辆生成器（自行车模型：巡航/变道/跟车/转弯/环岛）
- [x] 模型微调（QLoRA）+ 量化（GGUF Q4_K_M）
- [x] Orin 部署 + Demo（Gradio，agent 类型 + CVM 基线）
- [x] CVM 基线（车辆 + 行人）
- [x] 车辆模型重训 + 评估（合成：LLM ADE 1.58 vs CVM 7.41）
- [ ] 行人归一化重训评估（待 Orin 部署）
- [ ] 真实车辆（NGSIM）评估（待手动下载数据）

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

# 项目计划（定稿）：Alpamayo-R1 边缘量化部署（全 Orin · 零预算 · 零训练）

## 决策锁定
- **模型**：`nvidia/Alpamayo-R1-10B`（= Alpamayo 1）。**不用 1.5** —— TensorRT-Edge-LLM 的支持矩阵
  当前只有 `alpamayo_r1`（VLM 主干映射 `Qwen3VLForConditionalGeneration` + `AlpamayoAction` 头），
  1.5 尚未进边缘工具链。模型更强看 1.5，能落地看 R1，本项目选 R1。
- **量化/部署栈**：**NVIDIA TensorRT-Edge-LLM**（非 TensorRT-LLM）。它是面向 Physical AI 的边缘
  C++ LLM/VLM/**VLA** 运行时，**内置量化**（`tensorrt-edgellm-quantize`，ModelOpt 系）+ 对 Alpamayo
  一等公民支持（action 头 ONNX 导出 + `cpp/action/alpamayo1ActionRunner`）。
- **精度**：Orin 运行时仅支持 **FP16 / INT8 / INT4**（FP8/FP4/NVFP4 需 Blackwell）。→ 用
  **INT4 AWQ** 或 **INT8 SmoothQuant**。
- **硬件**：全程 **Jetson Orin AGX 64GB**（量化时装得下 10B BF16 ~20GB）。**零云、零训练、零预算。**
- **数据**：`nvidia/PhysicalAI-Autonomous-Vehicles`（门控、133TB）—— 只下几百段子集做真实评测。
- **许可**：权重非商用（研究/简历 OK），推理代码 Apache-2.0。

## 里程碑
| M | 目标 | 交付物 | 验收 | 机器 |
|---|---|---|---|---|
| **M0** 环境 | 装 TensorRT-Edge-LLM（Orin/JetPack），跑通 Quick Start，拉 Alpamayo-R1 权重；**pin 住 quantize/export 的确切 CLI 参数** | 环境可复现 + 单样本推理通 | `tensorrt-edgellm` 可用；R1 FP16 出 1 条轨迹 | Orin |
| **M1** 数据+口径 | 申请 PhysicalAI-AV，`physical_ai_av` 下 200–500 段 → `eval` JSONL（obs/gt，6.4s/64pt/10Hz） | 子集 + `data/prepare_physicalai_av.py` 完成 | 抽样可视化肉眼合理 | 本地/3070 |
| **M2** FP16 基线 | Orin 上 R1 FP16 推理，测 ADE/FDE/MR + 延迟；抽检推理痕迹 | 基线结果表（R1-FP16 vs CVM） | 全精度数值 + 延迟落地 | Orin |
| **M3** 量化 | `tensorrt-edgellm-quantize` 出 INT8、INT4 检查点 → export → build engine | 掉点曲线（FP16/INT8/INT4：体积/显存/精度） | ≥2 精度量化+测出 | Orin |
| **M4** 基准 | `benchmark.py` 测 延迟/吞吐/功耗(tegrastats)/显存，三精度对比 | Orin 基准表 | 端到端延迟测出 | Orin |
| **M5** 报告 | 架构/方法/结果/权衡/诚实局限 + 图表 + 可选 demo viewer | repo + 报告 + 简历 bullet | 可展示 | 本地 |

**关键路径**：M0（Orin 上 TensorRT-Edge-LLM 构建 + pin CLI）与 M1（数据审批）最可能拖期，可并行。

## 目录结构（本分支 `alpamayo-edge/`）
```
alpamayo-edge/
├── README.md
├── docs/plan.md                 # 本文件
├── configs/orin.env             # Orin 端点 / 模型 / 量化精度
├── data/
│   └── prepare_physicalai_av.py # PhysicalAI-AV → eval JSONL（M1 桩）
├── eval/                        # ← 从父仓库迁移，已本地测通
│   ├── metrics.py               # ADE/FDE/MR + 95%CI（迁移自 evaluate.py）
│   ├── baselines.py             # CVM / last-step CV（迁移自 baselines.py）
│   ├── evaluate.py              # 按 JSONL 打分（模型 or 基线）
│   └── viz.py                   # BEV 可视化（迁移自 demo/app.py）
├── scripts/
│   ├── quantize.sh              # tensorrt-edgellm-quantize（INT4/INT8）
│   ├── export_and_build.sh      # export + 边缘 build engine
│   └── benchmark.py             # 延迟/吞吐/功耗（tegrastats）骨架
├── checkpoints/ engines/ data/av_subset/ outputs/   # gitignore
```

## 复用清单（来自父仓库 qwen-trajectory-prediction）
| 迁移项 | 来源 | 状态 |
|---|---|---|
| ADE/FDE/MR + 置信区间 | `scripts/evaluation/evaluate.py` | ✅ 已迁移为 `eval/metrics.py`（改为吃轨迹数组，本地测通） |
| CVM 强基线 | `scripts/evaluation/baselines.py` | ✅ `eval/baselines.py`（数组版，测通） |
| 评估驱动 + 报告 | `evaluate.py` | ✅ `eval/evaluate.py`（JSONL，支持 `--baseline cvm`，测通） |
| BEV 可视化 + CJK 字体 | `demo/app.py` | ✅ `eval/viz.py`（obs/LLM/CVM/GT 四线，测通） |
| Orin 部署/重启套路 | `scripts/deploy/deploy_to_orin.sh` | 🔁 思路复用，引擎换 TensorRT-Edge-LLM |
| 方法学经验（带 CVM、诚实报负、断点续跑、env 配端点） | 全项目 | ✅ 继承 |
| GGUF/`quantize_gguf.py` | — | ❌ 不适用（改用 tensorrt-edgellm-quantize） |

## 风险与缓解
- **M0 Orin 构建摩擦**（aarch64/JetPack，"initial support"）→ 用官方预编译 wheel/容器;跑不通退回 PyTorch 推理 + 只测精度。
- **Alpamayo action 头导出**→ 官方已有 `alpamayo` 模块 + C++ runner,按其示例走;仍失败则只量化/部署 VLM 主干(Cosmos-Reason)。
- **数据审批延迟**→ 审批期先做 M0 + 用合成占位样本跑通 `eval/` 全链路。
- **Orin 上 10B+视频难实时**→ 价值转为"延迟/功耗刻画",不强求实时。
- **1.5 更强但没边缘支持**→ 作为"未来工作";若工具链更新支持 1.5 再切。

## 简历表达（填入实测数）
用 **NVIDIA TensorRT-Edge-LLM** 将 10B 自动驾驶 VLA(Alpamayo-R1,含 flow-matching 轨迹头)
**INT4 量化并部署于 Jetson Orin**,显存 −X% / 端到端延迟 Y ms / ADE 掉点 <Z%,建立含 CVM 强基线的
ADE/FDE/MissRate 评测(PhysicalAI-AV 真实子集)。技能:模型量化(INT4/INT8/ModelOpt)、边缘部署
(Jetson Orin/TensorRT-Edge-LLM)、多模态 VLA、自动驾驶轨迹预测、精度-延迟-功耗权衡。

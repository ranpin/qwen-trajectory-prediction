# 项目计划（终版 · Opt1 混合）：AV VLA 边缘量化与部署

> 零预算(仅用免费云做一次性重活) · 零训练 · 常态全 Orin。
> 背景与官方文档核实见 [FINDINGS.md](FINDINGS.md)。

## 决策锁定（最终）
- **两条模型轨道**:
  - **量化轨道** → `nvidia/Cosmos-Reason2-8B`(TensorRT-Edge-LLM 受支持的 VLM,同属 Physical-AI 家族):
    做 **INT4 AWQ / INT8 SmoothQuant**,产出真·精度-延迟-显存权衡曲线。
  - **VLA 轨道** → `nvidia/Alpamayo-R1-10B`(**FP16-only**,当前无量化):边缘部署做**自动驾驶轨迹预测** + CVM 基线评测。
- **栈**:NVIDIA **TensorRT-Edge-LLM** v0.9.0(内置量化 + Alpamayo action 头支持)。
- **设备**:仅 **Jetson Orin AGX 64GB**(运行时支持 FP16/INT8/INT4;Alpamayo-FP16 在 Orin 未经官方验证,列为风险)。
- **算力分工(关键)**:
  | 步骤 | 跑在哪 | 原因 |
  |---|---|---|
  | 数据子集 / 评测 / CVM / 可视化 | **本地 3070 / CPU** | 轻量,已就绪 |
  | **量化(8B)/ 导出(10B→ONNX)** | **免费云 Kaggle 2×T4=32GB**(一次性) | 8GB 3070 装不下 8B/10B |
  | build engine / 推理 / benchmark | **Orin** | 边缘部署本体 |
- **许可**:Alpamayo 权重非商用(研究/简历 OK);TensorRT-Edge-LLM Apache-2.0。
- **数据**:`nvidia/PhysicalAI-Autonomous-Vehicles`(门控,只下几百段子集)。

## 里程碑
| M | 目标 | 跑在哪 | 交付/验收 | 状态 |
|---|---|---|---|---|
| **M0** 环境 | Orin 装 TensorRT-Edge-LLM(JetPack 6.2+/7.2);免费云装量化/导出包;pin CLI | Orin + 云 | 两端可用;Quick Start 通 | ✅ 完成 |
| **M1** 数据+口径 | 申请 PhysicalAI-AV;下 200–500 段→eval JSONL(obs/gt,6.4s/64pt/10Hz);CVM 基线 | 本地 | 子集+CVM 数值 | ◻ 待数据审批（合成占位已通） |
| **M2a** 量化(Cosmos-Reason2-8B) | 云上 INT8/INT4 → export ONNX → scp → Orin build → **推理跑通** | 云→Orin | ≥2 精度引擎 | ✅ 完成（INT4+INT8，Orin 真实出 token） |
| **M2b** Alpamayo FP16 | 云上 export(onnx/llm+visual+action)→ scp → Orin build | 云→Orin | Orin 上出 1 条轨迹(验证 Orin 可跑) | ◻ 未启动 |
| **M3** 评测 | Cosmos 量化掉点曲线;Alpamayo 轨迹 ADE/FDE/MR vs CVM(经 action_to_traj 积分) | 本地+Orin+云 | 两张结果表 | ✅ 大部（Cosmos 掉点定量已出：FP16 参考下 PPL INT4 +9.8%/INT8 +18.1% + token 一致率，见 `eval/accuracy/`；仅 AV 轨迹表压在 VLA 轨道） |
| **M4** 基准 | Orin 延迟/吞吐/功耗(tegrastats):Cosmos FP16/INT8/INT4;Alpamayo FP16 | Orin | 基准表 | ◧ 大部（INT4/INT8 延迟+吞吐+显存+**功耗/能效+sweep+多模态+掉点**已测；仅缺 Orin 上 FP16 延迟点——FP16 仅云端跑了精度参考，未在 Orin 建引擎） |
| **M5** 报告 | 架构/方法/权衡/局限 + 图 + 简历 bullet | 本地 | 可展示 repo | ✅ 完成（[M5_report.md](M5_report.md) 一页总结 + 5 张图表（含掉点）+ 简历 bullet；仅缺 AV 轨道数据） |

**关键路径/风险点**:M2b(Alpamayo 能否在 Orin FP16 跑通)是最大不确定;跑不通则 VLA 轨道降级为"导出成功+Orin 运行受阻"如实记录,量化轨道(Cosmos)已**独立跑通并出指标**,项目已完整成立。

## 目标对照（初始目标 vs 实际达成，2026-07-21）

> 初始目标：用 TensorRT-Edge-LLM 把**自动驾驶 VLA** 量化部署到 Orin，做**精度-延迟-功耗**权衡评测（含 CVM 强基线）；原设想量化 Alpamayo-R1-10B 本身。

| 维度 | 状态 | 差异说明 |
|---|---|---|
| 量化 + 边缘部署（工程主线） | ✅ **超额** | INT4/INT8 双量化 Orin 端到端跑通，额外啃下 FMHA 崩溃根因 |
| 延迟/吞吐/显存/功耗/能效基准 | ✅ **超额** | 原只要基准表，实际含能效 tok/J、上下文 sweep、多模态 |
| 精度-延迟-功耗**权衡** | ✅ **三轴齐** | 延迟/功耗/显存 + **精度轴已补**（FP16 参考下量化掉点 PPL INT4 +9.8%/INT8 +18.1%，2026-07-22 补齐）；口径为总部署差距的相对退化，非纯量化误差 |
| 量化载体 | ⚠️ **改标的** | 原想量化 Alpamayo，但它**只支持 FP16 不能量化**（[FINDINGS.md](FINDINGS.md) F1，硬约束）→ 量化落在受支持的 Cosmos-Reason2-8B |
| **AV 轨迹预测（领域本职）** | ❌ **未达成** | 用模型出 6.4s/64 路点、算 ADE/FDE/MR vs CVM——完全没做，整体压在暂缓的 VLA 轨道(M2b)里 |
| CVM 强基线 | ◧ 仅基线侧 | CVM 在合成数据跑通，但无真实模型轨迹与之对比；真实数据(PhysicalAI-AV)待审批 |

**一句话**：**工程/系统目标（量化→边缘部署→性能功耗评测）已扎实达成甚至超额**；**领域/AV 目标（真实轨迹预测 + 真数据 vs CVM）基本未触及**，取决于是否启动 VLA 轨道。若定位为"展示边缘量化部署能力"——已完整有深度；若定位为"能预测轨迹的 AV 边缘系统"——还差 VLA 这半壁。

## 并行编排
- **可立刻并行**:①你 HF 申请 PhysicalAI-AV(异步审批);②我把 M1 数据/评测脚手架补成"合成占位可跑"(不等数据);③我整理 M0 的 Orin + 免费云安装命令清单。
- **审批/环境就绪后汇合**:M2a、M2b 可**并行**(两条模型轨道互不依赖),各自 云导出→Orin build→评测。

## 目录结构（`alpamayo-edge/`）
```
docs/{plan.md, FINDINGS.md, M0_setup.md, edge_deploy_status.md(结果报告), orin_build_notes.md(工程日志)}
configs/orin.env
eval/  metrics.py  baselines.py(CVM)  evaluate.py  viz.py(BEV)  action_to_traj.py(accel,κ→xy)   # 已本地测通
scripts/ quantize.sh(Cosmos-Reason2 INT4/INT8)  export_and_build.sh(Alpamayo FP16)
         run_action_inference.sh  benchmark.py
data/ prepare_physicalai_av.py(M1)  [+ make_synthetic_av.py 占位, 待补]
checkpoints/ engines/ data/av_subset/ outputs/   # gitignore
```

## 复用清单（父仓库 → 已迁移并测通）
ADE/FDE/MR+CI · CVM 基线 · 评估驱动 · BEV 可视化 —— 见 `eval/`。方法学经验(带 CVM 对照、诚实报负、
断点续跑、env 配端点)继承。GGUF 那套不适用(改 tensorrt-edgellm-quantize)。

## 风险与缓解
- **M2b Alpamayo-on-Orin 未验证** → 先小样本验证;失败则 VLA 轨道只交"导出+分析",量化轨道保底。
- **免费云配额/时长** → 导出/量化是一次性;分步 checkpoint,产物落盘即下载。
- **Orin build 摩擦(initial support)** → 用官方预编译 wheel/容器;失败退 PyTorch 推理只测精度。
- **数据审批延迟** → 合成占位样本先跑通全链路。

## 简历表达（填实测数）
用 **NVIDIA TensorRT-Edge-LLM** 在 **Jetson Orin** 端侧部署自动驾驶多模态栈:对 **Cosmos-Reason2-8B**
做 **INT4/INT8** 量化(显存 **−42%** / INT4 decode **30.9 tok/s @ ~40W** / 掉点 PPL **+9.8%(INT4)、+18.1%(INT8)** vs FP16),
给出 decode-选-INT4、prefill-选-INT8 的选型依据。(VLA 轨道:部署 10B Alpamayo-R1 含 flow-matching 轨迹头
+ CVM 强基线 ADE/FDE 评测——暂缓。)
技能:模型量化(INT4/INT8/AWQ/SmoothQuant/ModelOpt)、边缘部署(Jetson Orin/TensorRT-Edge-LLM)、多模态 VLM、量化精度评测。

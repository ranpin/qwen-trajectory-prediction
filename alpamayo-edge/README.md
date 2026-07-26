# Alpamayo-Edge：自动驾驶多模态模型的边缘量化部署

> 用 **NVIDIA TensorRT-Edge-LLM** 把自动驾驶多模态大模型量化并部署到 **Jetson Orin**，
> 做**精度-延迟-显存-功耗**权衡评测（含 CVM 强基线）。**零预算 · 零训练 · 全 Orin。**

## 现状（一句话）

**Cosmos-Reason2-8B 量化轨道已在 Orin 端到端跑通**：INT4(AWQ) + INT8(SmoothQuant) 两版完成
引擎构建→加载→prefill→decode→真实文本/图像生成，并出齐延迟/吞吐/显存/功耗/能效/上下文
sweep/多模态全套指标。详见 **[docs/edge_deploy_status.md](docs/edge_deploy_status.md)**。

- 📊 **选型结论**：decode 看 **INT4**（30.9 tok/s、能效高 64%、显存省 42%）；prefill 看 **INT8**（3252 tok/s、能效高 2.3×）。
- 🔧 **最大的坑已修**：Orin 构建漏传 `-DEMBEDDED_TARGET=jetson-orin` 会静默排除 sm_87 的 FMHA kernel、推理必崩——根因与修复见 [docs/orin_build_notes.md](docs/orin_build_notes.md)。
- 📝 **模型选型**：量化落在 **Cosmos-Reason2-8B**（TRTEdge 直接支持、可 T4 量化）；原计划 **Alpamayo-R1-10B** 因 FP16-only（不支持量化）+ 大模型 OOM 风险，作为 VLA 轨道**暂缓**（见 [docs/FINDINGS.md](docs/FINDINGS.md)）。

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/M5_report.md](docs/M5_report.md) | **一页总结（可展示）**：做了什么、核心图表、能用来干什么、诚实边界、简历 bullet |
| [docs/edge_deploy_status.md](docs/edge_deploy_status.md) | **结果报告**：链路、产物体积、性能/功耗/sweep/多模态全套指标、选型结论、next-steps 处置 |
| [docs/orin_build_notes.md](docs/orin_build_notes.md) | **工程日志**：正确 Orin 构建配方、复现命令、全部踩坑、FMHA 崩溃根因深挖 |
| [docs/orin_goat_migration_plan.md](docs/orin_goat_migration_plan.md) | **64GB orin-goat 迁移计划**：补 FP16 性能基线 + 跨设备复现（32GB 机 FP16 build OOM 的解法） |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | **方法与可复现性**：模型/权重来源、数据集、指标定义与测法、**量化校准集(cnn_dailymail 512)及影响分析** |
| [eval/accuracy/](eval/accuracy/) | 量化精度损失评测产物：prompts + Orin 输出 + FP16 参考 + 汇总脚本 |
| [docs/plan.md](docs/plan.md) | 项目计划、里程碑（M0–M5 状态）、目录结构、风险 |
| [docs/FINDINGS.md](docs/FINDINGS.md) | 选型背景（Alpamayo 只支持 FP16 等三条硬事实，已定案） |
| [docs/M0_setup.md](docs/M0_setup.md) | 两端环境搭建命令（导出主机 + Orin） |
| [cloud/README.md](cloud/README.md) | 免费云（Kaggle）一次性量化/导出 |
| [outputs/baseline_results.md](outputs/baseline_results.md) | CVM 评测基线（合成 AV 占位，待真实数据） |
| skill `edge-quantize-deploy` | 可复用的端到端工作流（`.claude/skills/`） |

## 快速复现（Orin，量化产物已在 Orin）

```bash
cd /home/vision/TensorRT-Edge-LLM
# 本机已把 build 软链到 build_orin 并在 ~/.bashrc 持久化 EDGELLM_PLUGIN_PATH，默认即正确插件；换环境时才需手动 export
# 性能
./build_orin/examples/llm/llm_bench --engineDir engines/int4/llm --mode prefill --inputLen 512 --iterations 5  --warmup 2
./build_orin/examples/llm/llm_bench --engineDir engines/int4/llm --mode decode  --pastKVLen 512 --iterations 20 --warmup 3
# 功能（VLM 需带视觉编码器）
./build_orin/examples/llm/llm_inference --engineDir engines/int4/llm \
  --multimodalEngineDir engines/int4/visual --inputFile cosmos_input.json --dumpOutput --maxGenerateLength 128
```
完整构建配方（含 `-DEMBEDDED_TARGET=jetson-orin`）与建引擎命令见 [docs/orin_build_notes.md](docs/orin_build_notes.md)。

## 评测轨道（AV 轨迹，本地）

CVM 基线 + ADE/FDE/MR 评测脚手架已从父仓库迁移并本地测通（`eval/`），当前用合成 AV 占位；
数据格式 `{"id":"clip","obs":[[x,y]],"gt":[[x,y]…64],"pred":[[x,y]…64]}`。接真实模型轨迹待 VLA 轨道启动。

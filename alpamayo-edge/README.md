# Alpamayo-Edge:自动驾驶 VLA 的边缘量化部署

> 用 **NVIDIA TensorRT-Edge-LLM** 把 10B 自动驾驶 VLA **Alpamayo-R1** INT4/INT8 量化并部署到
> **Jetson Orin**,做精度-延迟-功耗权衡评测(含 CVM 强基线)。**零预算 · 零训练 · 全 Orin。**

## 是什么 / 为什么
- **模型**:`nvidia/Alpamayo-R1-10B` —— VLA(Cosmos-Reason 主干 + flow-matching 轨迹头),
  输出 6.4s / 64 路点 / 10Hz 轨迹 + 因果链推理。选 R1 因为它是 TensorRT-Edge-LLM 当前支持的版本。
- **栈**:TensorRT-Edge-LLM(NVIDIA 官方边缘 LLM/VLM/VLA 运行时,内置量化)。
- **数据**:NVIDIA PhysicalAI-Autonomous-Vehicles(真实车辆,门控;只用小子集评测)。

## 快速开始(流程概览)
```bash
set -a && . configs/orin.env && set +a
# M1 数据(申请后):
python data/prepare_physicalai_av.py --out data/av_subset/test.jsonl --max_clips 300
# M3 量化(Orin 上):
bash scripts/quantize.sh                 # INT4 AWQ / INT8 SmoothQuant
bash scripts/export_and_build.sh         # export + 边缘 build engine
# 评测(本地即可,已测通):
python eval/evaluate.py --samples data/av_subset/test.jsonl --baseline cvm --label CVM -o outputs/cvm.json
python eval/evaluate.py --samples preds.jsonl --label "Alpamayo-R1 INT4" -o outputs/llm_int4.json
# M4 基准(Orin):
python scripts/benchmark.py --samples data/av_subset/test.jsonl --power -o outputs/bench_int4.json
```

## 现状
- ✅ 评测/CVM/可视化(`eval/`)已从父仓库迁移并**本地测通**(可用合成样本跑)。
- ✅ **端到端跑通,Orin 上真实出 token**:`nvidia/Cosmos-Reason2-8B` 的 **INT4(AWQ)** + **INT8(SmoothQuant)** 两版均在 Jetson Orin(sm_87)完成 引擎构建→加载→prefill→decode→连贯文本生成。详见 [docs/edge_deploy_status.md](docs/edge_deploy_status.md)。
- 📊 **性能对照**:INT4 decode 30.9 tok/s(交互式首选,显存 4.6GB);INT8 prefill 3252 tok/s(吞吐首选)。INT4 decode 快 1.57×、INT8 prefill 快 1.9×。
- 🔧 **FMHA 崩溃已修复**:根因是当初 Orin 构建 TRTEdge 漏传 `-DEMBEDDED_TARGET=jetson-orin`,CMake 默认按 sm_80;86;89 编译并 `-DEXCLUDE_SM_87` 排除了 sm_87 的 FMHA kernel;按官方 Orin 配方重编即通。与量化、模型均无关。
- 📝 **模型选型说明**:实际落地用 **Cosmos-Reason2-8B**(TRTEdge 直接支持、可 T4 上量化);原计划的 **Alpamayo-R1-10B** 因 FP16-only + 15GB 卡上易 OOM 暂缓。
- 🔜 M1 数据 / M4 基准 —— 见 [docs/plan.md](docs/plan.md)。

## 数据格式(eval JSONL)
```json
{"id":"clip","obs":[[x,y]],"gt":[[x,y]... 64],"pred":[[x,y]... 64]}
```
完整里程碑、目录、复用清单、风险见 **[docs/plan.md](docs/plan.md)**。

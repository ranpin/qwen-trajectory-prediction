# M5 — 项目总结报告（可展示版）

> 一页看懂：做了什么、拿到什么数、能用来干什么。详细指标见 [edge_deploy_status.md](edge_deploy_status.md)，
> 工程细节见 [orin_build_notes.md](orin_build_notes.md)，目标对照见 [plan.md](plan.md)。

## 做了什么

把 NVIDIA **Cosmos-Reason2-8B**（自动驾驶物理场景推理多模态大模型）用 **TensorRT-Edge-LLM**
量化为 **INT4 (AWQ)** 与 **INT8 (SmoothQuant)**，部署到 **Jetson Orin (sm_87)**，端到端跑通
**引擎构建 → 加载 → prefill → decode → 真实文本/图像推理**，并测出完整的延迟/吞吐/显存/功耗/能效画像。

- **输入**：图像/视频帧 + 文字问题（如"这个驾驶场景里有什么、危不危险"）。
- **输出**：文字推理（场景描述、风险判断、行动建议）——**不是轨迹路点**（那是 VLA/Alpamayo 的活）。
- 定位：AV 栈里的**"感知+推理/可解释"层的边缘部署**，不是路径规划层。

## 核心结果

![throughput](figures/bench_throughput.png)

![power & energy](figures/bench_power_energy.png)

![decode scaling](figures/decode_scaling.png)

![footprint](figures/footprint.png)

![accuracy drop-off](figures/accuracy_dropoff.png)

**选型结论**：**decode 看 INT4、prefill 看 INT8**——两者在不同阶段各自既更快又更省电；且精度上 **INT4(AWQ) 掉点更小**（vs FP16 perplexity +9.8% vs INT8 +18.1%）。

| 场景 | 选 | 理由 |
|---|---|---|
| 交互式 / 单流低延迟 / 省电 / 精度敏感 | **INT4** | decode 吞吐高 57%、能效高 64%、显存省 42%，掉点更小（+9.8%） |
| 批量 / 长上下文 / 高吞吐 | **INT8** | prefill 快 1.9×、能效高 2.3×、功耗更低（精度略逊，+18.1%） |

（环境：Orin sm_87 / JetPack 6 / CUDA 12.6 / TRT 10.7 / MAXN；batch=1，prefill 512 tok，decode pastKV=512。）

## 最硬的工程亮点：定位并修复 FMHA 崩溃

推理一进 attention 就 `There must be one kernel to implement the MHA` 崩溃。通过**读崩溃点源码 →
插桩打印运行时 kernel hashKey → 查编译宏 → 追到 CMake 默认分支**，定位真根因：**当初 Orin 构建
漏传 `-DEMBEDDED_TARGET=jetson-orin`，导致 CMake 按 sm_80;86;89 编译并 `-DEXCLUDE_SM_87` 把
sm_87 的 FMHA kernel 整段编译排除**。不是架构缺 cubin、不是模型、不是量化问题——是构建配置失误。
按官方 Orin 配方重编即通。（完整链路见 [orin_build_notes.md](orin_build_notes.md)。）

## 能用来干什么

1. **可复用的边缘量化部署方法论**：同一套流程可把任意 TRTEdge 支持的 LLM/VLM（Qwen-VL、其他 Cosmos）
   量化部署到任意 Jetson，并给出量化选型依据。已沉淀为 skill `edge-quantize-deploy`。
2. **一个可实时查询的车载场景推理系统**：喂驾驶图像 + 问题 → ~30 tok/s @ ~40W 的边缘推理。可作
   感知理解 / 安全监控 / 可解释性模块。
3. **一份完整的 INT4-vs-INT8 边缘权衡实证**：延迟/吞吐/显存/功耗/能效/上下文扩展性/**量化掉点** 全维度对照。

## 诚实的边界（未做 / 暂缓）

- **无轨迹预测（AV 本职任务）**：ADE/FDE/MR vs CVM 属于 VLA 轨道（Alpamayo-R1-10B FP16），大工程，暂缓。
- **精度轴已补齐但为相对退化口径**：有 FP16 参考下的 perplexity 掉点（INT4 +9.8% / INT8 +18.1%）与 token 一致率；该数字捕获"量化+引擎+后端"的总部署差距，非纯量化误差，也非下游任务正确率（无带标注的 AV QA 集）。详见 [edge_deploy_status.md](edge_deploy_status.md) 掉点节。
- 量化标的从 Alpamayo 改为 Cosmos，因 **Alpamayo 只支持 FP16、不能量化**（硬约束，见 [FINDINGS.md](FINDINGS.md)）。

## 简历 bullet（填实测数）

- 用 NVIDIA TensorRT-Edge-LLM 将 8B 多模态大模型 (Cosmos-Reason2) 量化为 INT4/INT8 并部署至 Jetson Orin，
  实测 **INT4 decode 30.9 tok/s @ ~40W (0.82 tok/J)**、**INT8 prefill 3252 tok/s (62.8 tok/J)**，
  并用 FP16 参考量化掉点（**INT4 perplexity +9.8% / INT8 +18.1%**），给出"decode 选 INT4、prefill 选 INT8"的选型依据。
- 定位并修复 sm_87 上的 FMHA kernel 派发崩溃：从崩溃断言追到 CMake 构建配置（漏传 Orin target flag
  致 sm_87 kernel 被编译排除），非平凡的跨层（运行时 kernel 表 ↔ 编译宏 ↔ CMake）根因排查。
- 建立 INT4/INT8 在延迟/吞吐/显存/功耗/能效/上下文扩展性/量化掉点的完整边缘画像，沉淀为可复用部署工作流。

> Resume (EN): *Quantized an 8B multimodal VLM (Cosmos-Reason2) to INT4/INT8 with NVIDIA
> TensorRT-Edge-LLM and deployed it to Jetson Orin; measured the full latency/throughput/memory/
> power/energy tradeoff (INT4 decode 30.9 tok/s @ ~40W; INT8 prefill 3252 tok/s) and quantization
> drop-off vs an FP16 reference (INT4 perplexity +9.8% vs INT8 +18.1%), and root-caused a hard
> sm_87 FMHA kernel-dispatch crash down to a missing CMake target flag.*

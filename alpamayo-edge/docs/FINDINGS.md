# ⚠️ 关键调研发现（会改动"定稿"方案）

来自 NVIDIA/TensorRT-Edge-LLM 官方文档(v0.9.0)的三条硬事实,直接影响上一版 plan:

## F1. Alpamayo 目前**只支持 FP16**,不支持 INT4/INT8
`examples/vla.md`:**"Only FP16 is supported for Alpamayo export in this release."**
→ 上一版"把 Alpamayo INT4/INT8 量化"的**核心卖点当前做不到**。量化工具
`tensorrt-edgellm-quantize` 面向**受支持的 LLM/VLM**(Qwen3、Qwen3-VL、Cosmos-Reason2…),
**不含 Alpamayo VLA**。

## F2. 官方 VLA 示例跑在 **Jetson Thor**,不是 Orin
`vla.md` 的 Build/Run 步骤标注 "**Thor Device**"。`installation.md` 里 Orin 明确支持
FP16/INT8/INT4 运行时,但**没有**演示 Alpamayo(VLA)在 Orin 上跑。
→ Alpamayo-FP16-on-Orin **理论可行**(FP16 在 Orin 支持集内、10B≈20GB 装得下 64GB),但**未被官方验证**,有风险。

## F3. 导出/量化必须在 **x86 + NVIDIA GPU 主机**上做
`installation.md`:"Quantization and `tensorrt_edgellm` ... **must run on an x86 Linux
system with an NVIDIA GPU**"。边缘设备只负责 build engine + 推理。
→ 你的 **x86 主机只有 RTX 3070 8GB**,导出 10B 模型可能吃紧(需 CPU offload 或更大主机)。

---

## 官方真实工作流(Alpamayo-R1,FP16)
1. **x86 主机**:`hf download nvidia/Alpamayo-R1-10B` → `tensorrt-edgellm-export MODEL MODEL/onnx --max-kv-cache-capacity 4096`(产出 onnx/{llm,visual,action})
2. **scp** onnx 到设备
3. **设备(Thor/Orin)**:`llm_build` + `visual_build` + `action_build`
4. **设备**:`action_inference --inputFile input_action.json`;输入=4相机×4时刻图像 + 轨迹历史[x,y,z];
   输出=`output_text`(推理)+ `output_trajectory`=**(accel, kappa) 对**(需 `eval/action_to_traj.py` 积分成 (x,y) 再算 ADE/FDE)

---

## 重新定标的可选方案(需你拍板)
| 方案 | 量化卖点 | AV/Alpamayo | 设备 | 可行性 |
|---|---|---|---|---|
| **Opt1 混合(推荐)** | 对**受支持模型 Cosmos-Reason2-8B** 做 INT4/INT8(真·量化曲线) | Alpamayo-R1 **FP16** 端侧部署 + 轨迹评测 | Orin(+x86 主机导出) | 量化端稳;Alpamayo-on-Orin 需验证 |
| Opt2 纯 Alpamayo | 无(FP16 only) | Alpamayo FP16 部署 + CVM 评测 | Thor 最稳 / Orin 待验 | 中;卖点弱在"量化" |
| Opt3 纯量化 | Cosmos-Reason2 / Qwen-VL INT4 on Orin | 无 AV | Orin | 最稳;丢了 AV 叙事 |

**建议 Opt1**:量化技能落在能量化的模型上,Alpamayo 作为"10B AV VLA 边缘 FP16 部署"。简历同时覆盖
量化 + 边缘 VLA + 自动驾驶轨迹。

## 需你回答
1. **设备**:你有 Jetson **Thor** 吗?还是**只有 Orin**?(决定 Alpamayo 部署风险)
2. **x86 导出主机**:只有 8GB 3070?能否用更大内存/显存的机器做一次性导出?
3. **方案**:Opt1 / Opt2 / Opt3?

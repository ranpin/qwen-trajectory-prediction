# PRD: 基于Qwen3-4B的轨迹预测系统

> 版本: v1.0 | 日期: 2026-06-16 | 作者: ranpin

---

## 一、项目概述

### 1.1 项目目标

构建一个完整的轨迹预测系统，基于Qwen3-4B大语言模型微调，实现从数据准备到NVIDIA Orin端侧部署的全链路闭环。

### 1.2 核心价值

- **LLM驱动轨迹预测**：利用大语言模型的语义理解能力，将轨迹预测建模为文本生成任务
- **可解释性**：模型不仅输出轨迹坐标，还能用自然语言解释预测理由
- **端侧部署**：通过量化加速，在NVIDIA Jetson Orin上实现实时推理
- **完整Demo**：提供交互式可视化演示系统

### 1.3 技术栈概览

| 环节 | 选型 | 说明 |
|------|------|------|
| 基座模型 | Qwen3-4B | 参数量适中，8GB显存可微调 |
| 数据集 | ETH/UCY + TrajNet++ + 合成数据 | 总量<200MB，适合有限算力 |
| 微调框架 | ms-swift | Qwen官方维护，原生支持LoRA |
| 微调方法 | QLoRA 4-bit | 8GB VRAM可行 |
| 量化 | AWQ 4-bit / GGUF Q4_K_M | 精度与速度平衡 |
| 推理引擎 | llama.cpp | Orin上最灵活，CPU+GPU混合推理 |
| 部署目标 | NVIDIA Jetson AGX Orin 64GB | 统一内存架构 |
| Demo框架 | Gradio Blocks | 交互式可视化 |

---

## 二、数据集方案

### 2.1 数据集选择

#### 实际使用情况

| 数据集 | 大小 | 状态 | 用途 |
|--------|------|------|------|
| 合成数据 | 51MB | ✅ 已生成50000条 | **主力训练数据** |
| ETH/UCY | <100MB | ⚠️ 远程服务器无法访问HF | 待手动下载后用于评估 |
| TrajNet++ | ~几十MB | ⚠️ 同上 | 待手动下载 |

#### 为什么先用合成数据？

1. **网络限制**：pc-3070远程服务器无法访问Hugging Face（网络不可达）
2. **快速验证**：合成数据50000条足够验证训练pipeline和模型效果
3. **后续扩展**：可以手动下载ETH/UCY后用于真实场景评估

### 2.2 数据序列化方案

将轨迹数据转换为LLM可处理的对话格式：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "你是一个专业的轨迹预测专家。根据给定的历史轨迹和场景信息，预测未来轨迹。"
    },
    {
      "role": "user",
      "content": "场景：十字路口\n行人A历史轨迹（过去4秒，每0.4秒采样）：\nt=0.0s: (2.3, 4.5)\nt=0.4s: (2.5, 4.8)\nt=0.8s: (2.7, 5.1)\nt=1.2s: (2.9, 5.4)\n速度：约0.8m/s，方向：东北\n\n请预测行人A未来3秒的轨迹。"
    },
    {
      "role": "assistant",
      "content": "根据行人A的运动趋势分析：\n- 速度保持稳定，约0.8m/s\n- 方向略微偏北，预计继续直线行走\n- 无明显加速或转向迹象\n\n预测轨迹：\nt=1.6s: (3.1, 5.7)\nt=2.0s: (3.3, 6.0)\nt=2.4s: (3.5, 6.3)\nt=2.8s: (3.7, 6.6)\nt=3.2s: (3.9, 6.9)"
    }
  ]
}
```

### 2.3 合成数据生成策略

使用Python脚本生成合成轨迹数据：

1. **基础运动模型**：
   - 匀速直线运动
   - 匀加速运动
   - 圆周运动（转弯）
   - 随机游走（行人）

2. **交互场景模拟**：
   - 跟车行为
   - 超车行为
   - 让行行为
   - 交叉冲突

3. **数据增强**：
   - 添加高斯噪声
   - 随机采样频率
   - 坐标系旋转
   - 时间偏移

4. **目标数据量**：10万-50万条轨迹对

---

## 三、模型微调方案

### 3.1 硬件配置

| 资源 | 规格 |
|------|------|
| GPU | RTX 3070 (8GB VRAM) |
| CPU | 8核+ |
| RAM | 32GB+ |
| 存储 | 500GB+ SSD |

### 3.2 微调框架：ms-swift

```bash
# 安装
pip install ms-swift -U

# QLoRA微调命令
swift sft \
    --model Qwen/Qwen3-4B \
    --tuner_type lora \
    --dataset ./data/processed/trajectory_sft.json \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --quantization_bit 4 \
    --torch_dtype bfloat16 \
    --output_dir ./outputs/qwen3-4b-trajectory-lora
```

### 3.3 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| lora_rank | 16 | LoRA秩，平衡容量和效率 |
| lora_alpha | 32 | 通常为rank的2倍 |
| learning_rate | 1e-4 | LoRA推荐学习率 |
| epochs | 3-5 | 防止过拟合 |
| batch_size | 1 | 8GB显存限制 |
| grad_accum | 16 | 有效batch=16 |
| max_length | 2048 | 轨迹文本长度 |
| quantization_bit | 4 | QLoRA 4-bit量化 |

### 3.4 训练策略

1. **阶段一：合成数据预训练**
   - 使用50万条合成轨迹
   - 训练3-5个epoch
   - 学习模型基础能力

2. **阶段二：真实数据微调**
   - 使用ETH/UCY + TrajNet++
   - 训练5-10个epoch
   - 学习真实分布

3. **阶段三：评估与调优**
   - 在ETH/UCY测试集评估
   - 根据minADE/minFDE调参
   - 必要时增加数据或epoch

### 3.5 评估指标

| 指标 | 定义 | 目标 |
|------|------|------|
| minADE | 最小平均位移误差 | <0.5m |
| minFDE | 最小终点位移误差 | <1.0m |
| Miss Rate | 终点误差>2m的比例 | <20% |
| 推理延迟 | 单次预测耗时 | <500ms |

---

## 四、模型量化与加速

### 4.1 量化方案对比

| 方法 | 比特 | 工具 | 优势 | 适用场景 |
|------|------|------|------|----------|
| AWQ | 4-bit | AutoAWQ | 精度损失最小 | Orin部署首选 |
| GPTQ | 4-bit | GPTQModel | 成熟稳定 | 备选方案 |
| GGUF | Q4_K_M | llama.cpp | CPU+GPU混合 | 内存受限场景 |

### 4.2 AWQ量化流程

```bash
# 安装
pip install autoawq

# 量化脚本
python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = './outputs/qwen3-4b-trajectory-lora/merged'
quant_path = './outputs/qwen3-4b-trajectory-awq'

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

quant_config = {'zero_point': True, 'q_group_size': 128, 'w_bit': 4}
model.quantize(tokenizer, quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
"
```

### 4.3 GGUF量化流程（llama.cpp）

```bash
# 克隆llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# 转换为GGUF格式
python convert_hf_to_gguf.py /path/to/model --outfile model-f16.gguf --outtype f16

# 量化为Q4_K_M
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

### 4.4 量化后模型规格

| 版本 | 大小 | 推理速度（估算） | 精度损失 |
|------|------|------------------|----------|
| FP16 | 8GB | 基准 | 0% |
| AWQ-4bit | 2GB | 2-3x加速 | <2% |
| GGUF-Q4_K_M | 2.3GB | 1.5-2x加速 | <3% |

---

## 五、NVIDIA Orin部署方案

### 5.1 目标硬件

**NVIDIA Jetson AGX Orin 64GB**

| 规格 | 参数 |
|------|------|
| AI算力 | 275 TOPS |
| 内存 | 64GB LPDDR5（统一内存） |
| 内存带宽 | 204.8 GB/s |
| GPU | 2048核 Ampere + 64 Tensor Core |
| CPU | 12核 ARM Cortex-A78AE |
| 功耗 | 15-60W可配置 |

### 5.2 部署架构

```
┌─────────────────────────────────────────────┐
│           NVIDIA Jetson AGX Orin            │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌──────────────┐    │
│  │ llama.cpp    │      │ Gradio Demo  │    │
│  │ (推理引擎)   │◄────►│ (Web界面)    │    │
│  └──────────────┘      └──────────────┘    │
│         │                                     │
│         ▼                                     │
│  ┌──────────────┐                           │
│  │ GGUF模型     │                           │
│  │ (Q4_K_M)     │                           │
│  └──────────────┘                           │
│                                             │
└─────────────────────────────────────────────┘
         ▲
         │ HTTP/WebSocket
         ▼
┌─────────────────────────────────────────────┐
│  外部客户端（PC/平板/手机）                  │
│  通过浏览器访问Demo                          │
└─────────────────────────────────────────────┘
```

### 5.3 llama.cpp部署步骤

```bash
# 1. 在Orin上安装llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUBLAS=1  # 启用CUDA加速

# 2. 传输量化模型
scp model-q4_k_m.gguf orin:/path/to/models/

# 3. 启动推理服务
./llama-server \
    -m model-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \  # 所有层offload到GPU
    -c 2048 \  # 上下文长度
    -t 8       # CPU线程数
```

### 5.4 性能优化策略

1. **GPU Offload**：使用 `-ngl 99` 将所有层加载到GPU
2. **内存管理**：Orin统一内存，无需担心VRAM不足
3. **批处理**：支持多请求并发（通过llama.cpp的server模式）
4. **功耗模式**：使用 `sudo nvpmodel -m 0` 设置为MAXN模式

### 5.5 部署验证

```bash
# 测试推理
curl http://orin-ip:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b-trajectory",
    "messages": [
      {"role": "user", "content": "场景：直行道\n车辆历史轨迹：..."}
    ],
    "max_tokens": 512
  }'
```

---

## 六、Demo演示系统

### 6.1 功能需求

1. **场景输入**：
   - 文本描述（场景类型、交通状况）
   - 历史轨迹输入（手动绘制或上传CSV）
   - 参数配置（预测时长、采样频率）

2. **可视化展示**：
   - 2D鸟瞰图（BEV）：历史轨迹 + 预测轨迹
   - 时间轴动画：逐帧展示轨迹演进
   - 多Agent展示：同时显示多个交通参与者

3. **模型输出**：
   - 预测轨迹坐标
   - 自然语言解释（为什么这样预测）
   - 置信度/不确定性估计

4. **性能监控**：
   - 推理延迟显示
   - 模型内存占用
   - GPU使用率

### 6.2 技术实现

```python
import gradio as gr
import matplotlib.pyplot as plt
import requests
import json

def predict_trajectory(scene_desc, history_traj, pred_horizon):
    """调用Orin上的llama.cpp服务进行推理"""
    prompt = f"场景：{scene_desc}\n历史轨迹：{history_traj}\n请预测未来{pred_horizon}秒的轨迹。"
    
    response = requests.post(
        "http://orin-ip:8080/v1/chat/completions",
        json={
            "model": "qwen3-4b-trajectory",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024
        }
    )
    result = response.json()["choices"][0]["message"]["content"]
    
    # 解析轨迹
    predicted_traj = parse_trajectory(result)
    
    # 生成可视化
    fig = plot_trajectory(history_traj, predicted_traj)
    
    return fig, result

# Gradio界面
with gr.Blocks(title="Qwen轨迹预测系统") as demo:
    gr.Markdown("# 🚗 基于Qwen3-4B的轨迹预测系统")
    gr.Markdown("### 微调 → 量化 → Orin部署 全链路Demo")
    
    with gr.Row():
        with gr.Column():
            scene_input = gr.Textbox(label="场景描述", placeholder="例如：十字路口，行人横穿马路")
            history_input = gr.Textbox(label="历史轨迹", placeholder="(x1,y1), (x2,y2), ...")
            horizon_slider = gr.Slider(1, 10, value=3, label="预测时长（秒）")
            predict_btn = gr.Button("预测", variant="primary")
        
        with gr.Column():
            plot_output = gr.Plot(label="轨迹可视化")
            text_output = gr.Textbox(label="模型输出", lines=10)
    
    predict_btn.click(
        predict_trajectory,
        inputs=[scene_input, history_input, horizon_slider],
        outputs=[plot_output, text_output]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
```

### 6.3 可视化效果

- **历史轨迹**：蓝色实线 + 圆点
- **预测轨迹**：红色虚线 + 三角形
- **地图元素**：车道线、路沿、交叉口（灰色）
- **Agent**：矩形框表示车辆/行人，带方向箭头

---

## 七、项目里程碑

### Phase 1: 数据准备（1-2周）

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| 下载ETH/UCY数据集 | data/raw/eth_ucpy/ | 5个场景数据完整 |
| 下载TrajNet++数据集 | data/raw/trajnetpp/ | JSON格式可解析 |
| 编写数据预处理脚本 | scripts/data_prep/preprocess.py | 输出统一格式 |
| 编写合成数据生成器 | scripts/data_prep/synthetic_gen.py | 可生成10万条数据 |
| 序列化为对话格式 | data/processed/trajectory_sft.json | 符合ms-swift要求 |

### Phase 2: 模型微调（2-3周）

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| 配置ms-swift环境 | requirements.txt | 可复现安装 |
| 合成数据预训练 | outputs/qwen3-4b-synthetic/ | loss收敛 |
| 真实数据微调 | outputs/qwen3-4b-trajectory-lora/ | minADE<0.8m |
| 模型合并 | outputs/qwen3-4b-merged/ | 可正常加载 |
| 评估报告 | docs/evaluation.md | 指标达标 |

### Phase 3: 量化加速（1周）

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| AWQ量化 | outputs/qwen3-4b-awq/ | 模型<3GB |
| GGUF量化 | outputs/qwen3-4b-q4_k_m.gguf | 可被llama.cpp加载 |
| 量化精度评估 | docs/quantization_eval.md | 精度损失<3% |
| 推理速度测试 | scripts/eval/benchmark.py | 延迟<500ms |

### Phase 4: Orin部署（1-2周）

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| Orin环境配置 | scripts/deploy/orin_setup.sh | llama.cpp可运行 |
| 模型部署 | Orin上的推理服务 | HTTP API可访问 |
| 性能优化 | 调优后的启动参数 | 延迟<300ms |
| 压力测试 | 并发请求测试 | 支持5+并发 |

### Phase 5: Demo开发（1-2周）

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| Gradio界面开发 | demo/app.py | 功能完整 |
| 可视化组件 | demo/visualizer.py | 轨迹图美观 |
| 集成测试 | 端到端测试 | Demo可运行 |
| 部署文档 | docs/demo_guide.md | 用户可复现 |

### Phase 6: 文档与演示（1周）

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| README编写 | README.md | 项目介绍清晰 |
| 技术文档 | docs/technical.md | 架构说明完整 |
| 演示视频 | demo/video.mp4 | 3-5分钟展示 |
| 开源发布 | GitHub Release | v1.0标签 |

**总工期估算：7-10周**

---

## 八、风险评估与缓解

### 8.1 技术风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 3070显存不足 | 中 | 高 | 使用QLoRA 4-bit，减小batch_size |
| 微调效果差 | 中 | 高 | 增加合成数据量，调整超参 |
| 量化精度损失大 | 低 | 中 | 对比AWQ/GPTQ/GGUF，选最优 |
| Orin推理延迟高 | 中 | 高 | 使用更激进的量化，或换小模型 |
| llama.cpp兼容性 | 低 | 中 | 备选TensorRT-LLM或MLC-LLM |

### 8.2 数据风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 合成数据不够真实 | 高 | 中 | 增加真实数据比例，改进生成模型 |
| ETH/UCY场景单一 | 中 | 中 | 后续扩展到HighD/Argoverse |
| 数据格式错误 | 低 | 低 | 编写数据校验脚本 |

### 8.3 部署风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| Orin硬件故障 | 低 | 高 | 云端GPU作为备选 |
| 网络延迟 | 中 | 中 | Demo和Orin部署在同一局域网 |
| 并发崩溃 | 中 | 中 | 限制最大并发数，添加队列 |

---

## 九、后续扩展方向

### 9.1 短期优化

1. **多模态输入**：集成摄像头图像，使用Qwen2.5-VL
2. **地图信息**：引入高精地图，提升预测准确性
3. **多Agent预测**：同时预测多个交通参与者的轨迹
4. **不确定性估计**：输出多条候选轨迹及置信度

### 9.2 中期目标

1. **更大数据集**：迁移到HighD或Argoverse 2
2. **更强模型**：升级到Qwen3-14B或Qwen3-30B-A3B
3. **闭环验证**：集成CARLA仿真器进行闭环测试
4. **实时推理**：优化到<100ms延迟，满足实时性要求

### 9.3 长期愿景

1. **车规级部署**：通过功能安全认证（ISO 26262）
2. **多传感器融合**：集成LiDAR、Radar等传感器数据
3. **在线学习**：支持模型在线更新，适应新场景
4. **商业化**：面向自动驾驶公司提供解决方案

---

## 十、附录

### 10.1 相关资源

- **Qwen3模型**：https://huggingface.co/Qwen
- **ms-swift框架**：https://github.com/modelscope/ms-swift
- **llama.cpp**：https://github.com/ggerganov/llama.cpp
- **ETH/UCY数据集**：通过Trajectron++获取
- **TrajNet++数据集**：https://github.com/vita-epfl/trajnetplusplusdata
- **Gradio文档**：https://gradio.app/docs

### 10.2 参考文献

1. GPT-Driver: Learning to Drive with Language Models (arXiv 2310.01415)
2. DriveLM: Driving with Graph Visual Question Answering (OpenDriveLab)
3. Qwen3 Technical Report (Alibaba)
4. ETH Pedestrian Dataset: "Learning Social Etiquette" (Pellegrini et al.)
5. TrajNet++: "Human Trajectory Prediction in Crowded Spaces"

### 10.3 术语表

| 术语 | 定义 |
|------|------|
| minADE | 最小平均位移误差（Minimum Average Displacement Error） |
| minFDE | 最小终点位移误差（Minimum Final Displacement Error） |
| LoRA | 低秩适配器（Low-Rank Adaptation） |
| QLoRA | 量化LoRA（Quantized LoRA） |
| AWQ | 激活感知权重量化（Activation-aware Weight Quantization） |
| GGUF | GPT-Generated Unified Format（llama.cpp模型格式） |
| BEV | 鸟瞰图（Bird's Eye View） |

---

**文档结束**

*最后更新: 2026-06-16*

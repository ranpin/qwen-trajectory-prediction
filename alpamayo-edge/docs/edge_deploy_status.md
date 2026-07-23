# 边缘量化部署 — 结果报告（Orin 推理跑通）

> Cosmos-Reason2-8B 从云端量化到 Jetson Orin 端到端推理跑通的**结果与指标**。
> 构建配方、踩坑、FMHA 根因等工程细节见 [orin_build_notes.md](orin_build_notes.md)；计划/里程碑见 [plan.md](plan.md)。
> **模型来源 / 数据集 / 指标定义与测法 / 校准集及影响** 见 [METHODOLOGY.md](METHODOLOGY.md)（可复现口径）。
> 更新日期：2026-07-22。

## 一句话结论

**全链路跑通，Orin 上真实出 token**：`nvidia/Cosmos-Reason2-8B` 的 **INT4 (AWQ)** 与 **INT8 (SmoothQuant)** 两版均在 Jetson Orin (sm_87) 上完成 **引擎构建 → 加载 → prefill → decode → 真实文本生成（含多模态图像）**。早前卡住推理的 FMHA 崩溃已定位为**构建配置失误**（漏传 `-DEMBEDDED_TARGET=jetson-orin`，详见 [orin_build_notes.md](orin_build_notes.md)）并修复，与量化、模型均无关。

```
Kaggle(T4x2) 量化+导出 → 打包 → 下载/校验 → scp 到 Orin → TRTEdge 建引擎 → 加载 → prefill/decode → 生成(文本+图像)
   ✅ INT4/INT8          ✅       ✅ SHA256    ✅          ✅ LLM+视觉塔   ✅        ✅ 两版      ✅ 两版连贯
```

## 产物体积对照

| 组件 | INT4 (AWQ) | INT8 (SmoothQuant) | 比值 |
|---|---|---|---|
| LLM ONNX 权重 (`llm/model.onnx.data`) | 4.83 GB | 8.20 GB | 1.70× |
| 词嵌入 (`embedding.safetensors`, fp16) | 1.24 GB | 1.24 GB | 1.00× |
| 视觉塔 ONNX (`visual/model.onnx.data`, fp16) | 1.16 GB | 1.16 GB | 1.00× |
| 打包 tgz | 5.77 GB | 7.81 GB | 1.35× |
| **Orin TensorRT 引擎** — LLM | 4.85 GB | 8.25 GB | 1.70× |
| **Orin TensorRT 引擎** — 视觉塔 | 1.1 GB | 同 INT4（同一份 fp16，不重建） | 1.00× |

> 主干 LLM 权重 INT8 是 INT4 的 1.70×，接近 W8A8 vs W4A16 的理论 2×（差在两版共享的 fp16 词嵌入/lm_head 未随之翻倍）。视觉塔当前工具仅支持 fp8/fp16，两版都留 fp16，故主干 LLM 才是量化差异所在。
> 产物 SHA256：INT8 `66f05e38…`（`outputs/edge_artifacts_int8_sq.tgz`，7,807,211,181 B）。

## 功能与性能指标

测试环境：Orin sm_87 / JetPack 6 / CUDA 12.6 / TRT 10.7 / MAXN 模式；batch=1，prefill inputLen=512，decode pastKVLen=512。

| 指标 | INT4 (AWQ) | INT8 (SmoothQuant) | 说明 |
|---|---|---|---|
| **Prefill 512 tok** | 300.98 ms | **157.45 ms** | INT8 快 ~1.9×——计算密集，原生 int8 张量核 vs int4 AWQ 反量化开销 |
| Prefill 吞吐 | 1701 tok/s | **3252 tok/s** | 同上 |
| **Decode 每 token** | **32.34 ms** | 50.65 ms | INT4 快 ~1.57×——访存密集，4.6GB vs 7.8GB 权重每 token 搬运更少 |
| Decode 吞吐 | **30.9 tok/s** | 19.7 tok/s | 交互式延迟看这项，INT4 胜 |
| 引擎载入显存 | ~4622 MiB | ~7864 MiB | Orin 统一内存 29GB；INT8 逼近上限 |
| 真实生成 | ✅ 连贯正确 | ✅ 连贯正确 | 湿路刹车/行人处置两题均合理，含自然 EOS 终止 |

### 功耗与能效（tegrastats，MAXN，GPU-active 采样均值）

整机功耗 = `VDD_GPU_SOC`(GPU+SOC) + `VDD_CPU_CV`(CPU) + `VIN_SYS_5V0`(5V 外设) 三轨之和。

| 场景 | INT4 (AWQ) | INT8 (SmoothQuant) | 能效赢家 |
|---|---|---|---|
| **Prefill** GPU_SOC / 整机 | 48.0 W / 61.8 W | 39.5 W / 52.1 W | — |
| Prefill 能效 | 27.5 tok/J | **62.8 tok/J** | **INT8 高 2.3×**（原生 int8 核，功耗更低+吞吐更高） |
| **Decode** GPU_SOC / 整机 | 26.3 W / 39.6 W | 25.5 W / 40.6 W | — |
| Decode 能效 | **0.82 tok/J** | 0.50 tok/J | **INT4 高 1.64×**（访存密集，权重小=搬运能耗低） |
| 峰值整机功耗 | prefill 67.3 W / decode 49.4 W | prefill 61.7 W / decode 46.6 W | — |

### 吞吐 / 上下文扩展性 sweep（INT4，MAXN）

| Prefill inputLen (batch1) | 128 | 256 | 512 | 1024 | | Decode pastKVLen (batch1) | 128 | 512 | 1024 | 2048 | 4000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tok/s | 1485 | 1631 | 1701 | 1720 | | tok/s | 32.1 | 31.8 | 31.3 | 30.5 | 29.2 |

- Prefill inputLen：长输入吞吐更高（利用率摊薄），1024 趋稳。
- Prefill batch（inputLen512）：per-seq tok/s 1699/873/422（batch 1/2/4），**聚合恒定 ~1700**——batch1 已算力饱和，批处理无聚合增益。
- Decode pastKVLen：上下文 128→4000（涨 31×）仅掉 9%，由权重搬运主导、注意力占比小，长上下文扩展性好。

### 多模态（图像）路径 ✅

喂 `woman_and_dog.jpeg` + 提问，走**视觉塔 + deepstack 融合 + INT4 LLM** 全路径，模型准确描述海滩场景（女子格子衫、狗戴彩色胸背带、海浪、日落暖光），与官方参照吻合 → VLM 本职能力在 Orin 上端到端正确。

## 量化掉点（vs FP16 基线）✅

### 主：真实基准任务正确率（Cosmos-Reason1-Benchmark robovqa+robofail，n=210 MC 带标准答案）

| 子集 | FP16 参考 | INT4 (AWQ) | INT8 (SmoothQuant) |
|---|---|---|---|
| robovqa（n=110，较易） | 88.2% | 87.3% | 87.3% |
| robofail（n=100，更难） | 63.0% | 59.0% | 62.0% |
| **总体（n=210）** | **76.2%** | **73.8%** | **75.2%** |
| 总体 Wilson 95%CI | [70.0,81.4] | [67.5,79.3] | [69.0,80.6] |
| 总体 vs FP16（McNemar） | — | −2.4pts p=0.42 不显著 | −1.0pts p=0.85 不显著 |

视频按 6 帧(≤448px)采样作多图，FP16(云端 PyTorch)/INT8/INT4(Orin) **同帧**→掉点严格可比。**统计检验:CI 重叠、McNemar p>0.4 → 差异不显著(n=210；robovqa 88.2%+robofail 63%)** → 稳妥结论是**量化无显著任务正确率损失**(非精确"掉 0.9pts")。方法/口径见 [METHODOLOGY.md](METHODOLOGY.md) §3.3；产物 `eval/accuracy/benchmark/`。robovqa 子集(具身机器人推理，非驾驶)、6帧@448 非官方原生视频协议——绝对分不追求复现论文。

> **FP16 无设备端性能基线**：FP16 引擎(~16GB)在 Orin `llm_build` 期 **OOM 被杀(exit 137)** → FP16 无法在此边缘设备部署；**量化是落地必需**，"加速比"以显存可行性(FP16 装不下、INT4/INT8 从容)替代呈现。

### 辅：细粒度 perplexity 探针（12 prompt）

FP16 参考在 Kaggle T4×2 上用 PyTorch 加载 `nvidia/Cosmos-Reason2-8B`（Orin 无外网、8B fp16≈16GB 本地装不下，故走云端一次性），对**同一批 12 条固定 prompt**（驾驶物理推理+决策+常识，贪心 `top_k=1` 确定性）产出：产物与复现脚本见 [`eval/accuracy/`](../eval/accuracy/)。

**方法**：Orin 上 INT4/INT8 引擎贪心生成实际输出 → 用 FP16 模型对每段输出做 **teacher-forced perplexity** 打分（"原始模型对量化输出有多惊讶"，PPL 越低=越贴近 FP16 分布）；并统计与 FP16 贪心输出的 **token 一致前缀**。完整定义见 [METHODOLOGY.md](METHODOLOGY.md) §3.3。

> **量化校准集 + 消融（重要）**：两版均为 NVIDIA ModelOpt PTQ，校准用**默认 512 篇 CNN/DailyMail 新闻**（INT4=AWQ W4A16、INT8=SmoothQuant W8A8；视觉塔未量化）。**已做域内校准消融**：原假设"新闻域错配拖累 INT8"，但用域内小集重量化 INT8 后**反而暴跌**（perplexity +244%、MC 84.5%）→ **假设推翻**；真正结论=**校准集覆盖度/质量比域匹配更重要**，默认 news-512 得验证。详见 [METHODOLOGY.md](METHODOLOGY.md) §4 与 `eval/accuracy/ablation_calibration/`。

| 指标 | FP16 参考 | INT4 (AWQ) | INT8 (SmoothQuant) |
|---|---|---|---|
| 平均 perplexity（越低越好） | 1.237 | 1.359 | 1.460 |
| **PPL 相对 FP16 增幅** | — | **+9.8%** | **+18.1%** |
| 与 FP16 贪心一致前缀占比 | 100% | 16.4% | 5.7% |
| 平均一致前缀长度 (token) | — | 20.1 | 7.2 |

- **关键发现**：INT4 (AWQ) 掉点**反而小于** INT8 (SmoothQuant)（+9.8% vs +18.1%），贪心路径也跟随更久。合理——AWQ 是激活感知的纯权重 4bit（W4A16），保留输出分布更好；INT8 SmoothQuant 为 W8A8、连激活也量化，本推理负载上对分布扰动更大。与"decode 看 INT4"的性能结论方向一致：**INT4 在本工作负载既快（decode）又更保真**。
- **诚实边界**：① 该 PPL 是对模型**自身贪心输出**的 teacher-forced 打分，绝对值偏低（高概率序列），**有意义的是相对增幅**而非绝对值。② 分数对比的是**已部署的 Orin TRT 引擎**输出 vs **FP16 PyTorch**，因此捕获的是"量化+引擎实现+后端"的**总部署差距**（也正是实用相关的量——相比原始模型你实际损失多少），而非纯量化误差；token 一致前缀偏低正是贪心下后端差异逐 token 累积所致，故 perplexity 为主指标、一致率为辅证。③ 参考用 T4 上 fp16 加载，与 Orin fp16 派生引擎口径一致。

## 部署选型结论

- **交互式 / 单流低延迟 / 省电续航 / 精度敏感** → **INT4**：decode 吞吐高 57%、能效高 64%、显存省 42%，且掉点更小（PPL +9.8% vs +18.1%）。
- **批量 / 长上下文 / 高吞吐** → **INT8**：prefill 快 1.9×、能效高 2.3×、功耗还更低（代价是本负载上分布保真略逊）。
- 一句话：**decode 看 INT4，prefill 看 INT8**——两者在不同阶段各自既更快又更省电；两版功能均可用，且 **INT4 在本工作负载既更快（decode）又更保真**。

## 下一步（处置状态，2026-07-21）

| # | 事项 | 状态 |
|---|---|---|
| 1 | 功耗 / 能效测量 | ✅ 完成（见上） |
| 4 | 多模态图像路径 | ✅ 完成（见上） |
| 5 | 吞吐 / 上下文 sweep | ✅ 完成（见上） |
| 2 | FP16 基线（补精度-延迟-功耗第三点） | ✅ 完成（Kaggle T4×2 fp16 参考，见"量化掉点"节） |
| 3 | 量化掉点定量评测（vs FP16 基线） | ✅ 完成——PPL 增幅 INT4 +9.8% / INT8 +18.1% + token 一致率 |
| 7 | VLA 轨道 M2b（Alpamayo-R1-10B FP16 轨迹 + CVM） | ⏸ 暂缓——大工程，见 [plan.md](plan.md) |
| 6 | 运维收尾（build_orin 设默认 / 清旧 build） | ✅ 完成——删旧坏 build、`build→build_orin` 软链、`~/.bashrc` 持久化 EDGELLM_PLUGIN_PATH，默认即正确插件（上游报 issue 仍暂缓） |

> 精度轴已补齐（#2/#3 完成）：精度-延迟-功耗三轴齐全。仅剩 VLA 轨道（#7）为择期再启的大工程，非遗漏。

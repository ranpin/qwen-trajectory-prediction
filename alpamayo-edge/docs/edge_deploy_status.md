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

> **FP16 基线（经 64GB orin-goat 补全）+ 量化加速比**：FP16 引擎在 32GB dog `llm_build` 期 OOM（**峰值实测 54.8GB**，远超 30GB；运行时激活仅 8MB → 是 build 内存墙、非推理）。改在 **64GB orin-goat** build+跑得同机三方：TTFT FP16 298 / INT8 159 / INT4 305 ms，TPOT FP16 89.3 / INT8 52.4 / INT4 33.6 ms。**decode 加速 vs FP16：INT4 2.66×、INT8 1.70×；prefill：INT8 1.87×、INT4≈FP16；E2E(512+128) 11.7→4.6s(2.5×)**。**实测:FP16 引擎(15GB)在 32GB dog 连载入都 OOM**(deserialize 申请 15GB 时仅 ~11GB free,引擎文件页缓存吃满)→ FP16 build 与 load 在 32GB 上双双不可行,只在 64GB goat 跑;dog 部署 INT4/INT8。跨设备 INT4/INT8 两机 <3% 已复现。**能效(goat 同机,tok/J)**:decode FP16 0.42/INT8 0.90/INT4 1.31 → INT4 decode 能效比 FP16 高 **3.1×**;prefill FP16 30.1/INT8 62.0/INT4 26.9。详见 `eval/perf/fp16_speedup_goat.json`、[orin_goat_migration_plan.md](orin_goat_migration_plan.md)。

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

## Roofline 分析（2026-07-25 新增）

把实测工作点投到 Orin 的 roofline（`eval/roofline.py` → `docs/figures/roofline.png`）上，三个反常点都有了定量解释：

| 精度 | Decode AI | Decode 实测带宽（占 204.8 GB/s） | Prefill AI | Prefill 实测吞吐（占天花板） | 天花板 |
|---|---|---|---|---|---|
| FP16 | 1.00 | 169.7 GB/s（**82.9%**） | 512 | 26.0 TFLOP/s（60.5%） | 43 TFLOP/s |
| INT8 (SQ) | 1.83 | 158.3 GB/s（77.3%） | 935 | 48.7 TOP/s（57.3%） | **85 TOP/s** 原生 int8 |
| INT4 (AWQ) | **3.16** | 142.8 GB/s（69.7%） | 1616 | 25.5 TFLOP/s（59.2%） | 43 TFLOP/s（**反量化回 FP16**） |

- 脊点 = 43 TFLOP/s ÷ 204.8 GB/s ≈ **210 FLOP/byte**。decode 的 AI 只有 1.0–3.2，**低于脊点 67–210×** → 纯访存墙。
- ⇒ **INT4 decode 的加速上限就是字节比 15.15/4.8 = 3.16×**；实测 2.66× 的缺口来自带宽效率随权重变小而降（82.9% → 69.7%，固定开销占比上升）。**decode 已达峰值带宽 70–83%，几无 kernel 优化空间**，要更快只能继续压字节（更低位宽 / KV cache 量化 / 稀疏）。
- prefill 的 AI 远高于脊点 → 计算墙。**W4A16 的天花板是 FP16 的 43 TFLOP/s 而非 INT8 的 85 TOP/s**（TensorRT WoQ 先反量化再做高精度点积，官方文档明示），所以 **INT4 prefill 结构上不可能快过 FP16**（25.5 vs 26.0 TFLOP/s，实测吻合）；INT8(W8A8) 用原生 int8 核、天花板翻倍，才有 1.87×。
- ⇒ 想让 INT4 的 prefill 也提速，必须上 **W4A8** 之类"激活也量化"的方案，让 GEMM 真跑在低精度张量核上。
- 口径：算力天花板均为 **GPU 稠密**值、**不含 DLA**；NVIDIA 宣传的 275 TOPS = GPU 稀疏 170 + 双 DLA 稀疏 105，GPU-only 稠密仅为其 31%。模型侧 7.575 B 矩阵乘参数由 config 解析推出，×2 B = 15.15 GB 与实测 FP16 引擎大小自校验一致。

## Kernel 级剖析（Nsight Systems，2026-07-25 新增）

复现命令与产物：`eval/profile/README.md`、`eval/profile/int4_decode_kern_sum.csv`；图 `docs/figures/kernel_breakdown.png`。
**方法坑（重要）**：TRTEdge 的 decode 走已捕获的 **CUDA graph**，nsys 默认 `--cuda-graph-trace=graph` 会把 20 次计时迭代折叠成一个区间，只有 3 次慢速 warmup 被逐 kernel 归因 → 份额和带宽都算错。**必须加 `--cuda-graph-trace=node`**（node 口径四项合计 33.8 ms ≈ 实测 TPOT 33.1 ms，自洽）。
**`ncu` 不可用**：Nsight Compute 硬件计数器需 root，本机 `sudo` 需密码 → 带宽改用「解析字节 ÷ 实测 kernel 耗时」，字节数已与引擎文件大小自校验到 **+0.13%**。

| kernel | 每 token 耗时 | 占比 | 有效带宽（占 204.8 GB/s 峰值） |
|---|---|---|---|
| **W4A16 GEMV**（每层 7 次：q/k/v/o + gate/up/down） | 24.39 ms | **72.1%** | **147.9 GB/s（72.2%）** |
| **lm_head fp16 GEMM**（N=151936） | 7.46 ms | **22.0%** | 166.9 GB/s（81.5%） |
| attention `kernel_mha` | 0.91 ms | 2.7% | — |
| 已融合 elementwise（RMSNorm/RoPE/SwiGLU） | 1.08 ms | 3.2% | — |

**引擎字节按角色分解**（合计 4.853 GB vs 实测引擎文件 4.847 GB，+0.13%）：INT4 权重 3.473 GB(71.6%)、AWQ scales+zeros 0.136 GB(2.8%)、**lm_head fp16（未量化）1.245 GB(25.6%)**。

三点发现：
1. **lm_head 根本没被量化**——AWQ 默认跳过输出层，于是它独占 **25.6% 的 decode 字节**；且 M=1 却被调度到 `tilesize64x96` 的 GEMM kernel（身份由 grid 维度确认：`grid=(1,1583,1)`，1583×96=151968≈vocab 151936）。
2. **效率最低的恰是最大头**：GEMV 只到峰值 72.2%，而 lm_head 有 81.5% → 手写 kernel 的目标空间在 GEMV。
3. **TRT 已把 RMSNorm/RoPE/SwiGLU 融合掉了**（kernel 名 `__myl_AddCasMulMeaAddSqrDivMulCasMulMulMul` 即融合链）且 CUDA graph 已启用 ⇒ **"算子融合 + 降 launch 开销"这条常规路线在此几无空间（合计仅 3.2%）——诚实的负结果**，原计划的 K4 据此重定向。

**由此推出的优化余量（预测，尚未实施）**：

| 优化 | 字节 | 预测 TPOT / 吞吐 | 代价 |
|---|---|---|---|
| A. lm_head 也量化到 INT4 | 4.853 → **3.932 GB**（−19%） | 33.1 → **≈26.8 ms**，30.2 → **≈37.3 tok/s（+23%）** | 输出层对量化敏感，需实测掉点 |
| B. 手写 W4A16 GEMV，带宽 72.2%→85% | 不变 | 33.1 → ≈30.2 ms（**+10%**） | 工程量；需做成 TRT plugin 才端到端生效 |
| A+B | −19% | **≈24.5 ms（+35%）** | — |

## 手写 kernel 对照与「优化 B」的推翻（K2，2026-07-25 新增）

产物 `eval/kernels/`（`w4a16_gemv.cu` + `results.csv` + README），图 `docs/figures/gemv_kernel_bench.png`。

先测**这台机器的真实可达读带宽**（自写 1GB 流式读 kernel，128-bit load）：**152.3 GB/s = 理论峰值 204.8 的 74.4%**
（两次运行 151.7 / 152.3，差 0.4%；与 NVIDIA 论坛社区实测 151.7 GB/s D2D 独立吻合）。

| | 有效带宽 | 占理论峰值 | 占**可达**带宽 |
|---|---|---|---|
| 实测可达读带宽 | **152.3 GB/s** | 74.4% | 100% |
| TRT-Edge-LLM `gemv_kernel` | 147.9 GB/s | 72.2% | **97.1%** |
| 手写 v4（x 入 shared memory + half2） | 79.2 GB/s | 39% | 52.0% |
| 手写 v1（标量 32-bit load） | 56.9 GB/s | 28% | 37% |
| 手写 v2/v3（128-bit load，无 shared x） | ~20 GB/s | 10% | 13% |

**⇒ 上一节「优化 B：手写 GEMV 提升带宽效率 → +10%」被实测推翻**：那个预测用了错的分母（理论峰值）。
按可达带宽算，TRT 的 GEMV 已达 **97.1%**，最多只剩 3.0%（端到端 ≈ **+2%**）⇒ **手写 kernel 不值得做**。
decode 真正的杠杆只剩**减少字节数**（优化 A：量化 lm_head，预测 +23%）。

三条工程教训：
1. **优化前先量天花板**，否则会对着理论峰值虚构出 28% 的余量。
2. **v2/v3 把 32-bit load 换成 128-bit 反而更慢**——瓶颈不是 DRAM，而是逐元素重复读 `x` 与标量 int→fp16 转换；
   v4 把 x 放进 shared memory（block 内 8 warp 复用）+ half2 后才快 1.4–4×。与 TRT 仍差 1.87×，
   补齐需 AWQ/FT 的 `lop3` 位技巧（直接拼 fp16 尾数免转换）+ 权重离线置换（TRT 已做），但因结论 1 不再投入。
3. **Jetson 按负载调 GPU/EMC 频率、`jetson_clocks` 需 root（不可用）**：计时前须持续预热（本基准跑 400 次）再取最优；
   否则短脉冲测得的天花板只有 104.9 GB/s（偏低 30%，首版就踩了这个坑）。

## 视觉塔剖析与多相机扩展性（2026-07-25 新增）

产物 `eval/vision/`（RESULTS.json + vision_viz.py），图 `docs/figures/vision_tower.png`。
方法：`llm_inference --dumpProfile --warmup 1`，取 TRT 自报的分段 GPU 时间；帧取 robovqa_0_*.jpg（≤448px）。

| 帧数 | image token | 视觉塔 (ms) | prefill token | prefill (ms) | TTFT (ms) | 视觉塔占比 |
|---|---|---|---|---|---|---|
| 1 | 112 | 29.05 | 189 | 121.58 | 150.6 | 19.3% |
| 2 | 224 | 51.33 | 303 | 195.96 | 247.3 | 20.8% |
| 4 | 448 | 93.52 | 531 | 336.43 | 429.9 | 21.8% |
| 6 | 672 | 146.17 | 759 | 446.41 | 592.6 | **24.7%** |

- **视觉塔耗时严格线性**：拟合 **4.5 + 23.2 ms × 帧**（R²=0.997），每帧 112 token。外推 8 路相机：视觉 ≈190 ms、TTFT ≈775 ms。
- **视觉塔是被忽略的 1/4，且是唯一仍是 fp16 的部件**（引擎 1.168 GB，ModelOpt 量化默认跳过它）→ 明确的下一步收益点。
- **发现硬约束**：8 帧 = 8×112+77 = **973 token，逼近引擎 `maxInputLen=1024`；9 帧装不下**。AV 的 6–8 路相机正好卡上限，每路多帧（视频）需用更大 maxInputLen 重建引擎。
- prefill 增长更快（≈65 ms/帧）⇒ **多相机场景瓶颈在 prefill，不在视觉塔本身**。
- **INT8 视觉塔本地不可行（实测确认）**：`visual_build` 无精度参数（仅 onnxDir/engineDir/token 上限），精度由 ONNX 决定 ⇒ 必须回云端重导 → 待凭据轮换后执行。
- **踩坑**：`~/.bashrc` 对非交互式 shell 提前 `return`，`EDGELLM_PLUGIN_PATH` 不生效；`llm_bench` 之前能跑是因 cwd 在 `TensorRT-Edge-LLM/` 下命中相对路径 `build/`（软链）。换 cwd 后引擎**反序列化直接失败**，须显式 export（已进 PROBLEMS.md）。
- **口径**：`llm_inference` 报的 generation "16.3 ms/token" **不采用**——它把单个 decode step 的 32.93 ms 除以 "Generated Tokens: 2"（其一由 prefill 产出）；32.93 ms 与 llm_bench 的 33.13 ms 吻合，TPOT 仍以 llm_bench 为准。Peak unified memory 恒为 ~4856 MB（≈LLM 引擎大小），似未计入另载的视觉塔引擎，原值记录、口径未确认。

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

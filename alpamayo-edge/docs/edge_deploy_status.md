# 边缘量化部署 — 结果报告（Orin 推理跑通）

> Cosmos-Reason2-8B 从云端量化到 Jetson Orin 端到端推理跑通的**结果与指标**。
> 构建配方、踩坑、FMHA 根因等工程细节见 [orin_build_notes.md](orin_build_notes.md)；计划/里程碑见 [plan.md](plan.md)。
> 更新日期：2026-07-21。

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

## 部署选型结论

- **交互式 / 单流低延迟 / 省电续航** → **INT4**：decode 吞吐高 57%、能效高 64%、显存省 42%。
- **批量 / 长上下文 / 高吞吐** → **INT8**：prefill 快 1.9×、能效高 2.3×、功耗还更低。
- 一句话：**decode 看 INT4，prefill 看 INT8**——两者在不同阶段各自既更快又更省电；功能上两版质量均可用、无明显退化。

## 下一步（处置状态，2026-07-21）

| # | 事项 | 状态 |
|---|---|---|
| 1 | 功耗 / 能效测量 | ✅ 完成（见上） |
| 4 | 多模态图像路径 | ✅ 完成（见上） |
| 5 | 吞吐 / 上下文 sweep | ✅ 完成（见上） |
| 2 | FP16 基线（补精度-延迟-功耗第三点） | ⏸ 暂缓——需再跑 Kaggle 导出，绑定凭据轮换 |
| 3 | 量化掉点定量评测（vs FP16 基线） | ⏸ 暂缓——依赖 #2 |
| 7 | VLA 轨道 M2b（Alpamayo-R1-10B FP16 轨迹 + CVM） | ⏸ 暂缓——大工程，见 [plan.md](plan.md) |
| 6 | 运维收尾（build_orin 设默认 / 清旧 build） | ✅ 完成——删旧坏 build、`build→build_orin` 软链、`~/.bashrc` 持久化 EDGELLM_PLUGIN_PATH，默认即正确插件（上游报 issue 仍暂缓） |

> 暂缓项均为用户决策的择期再启，非遗漏。#2/#3 是解锁"精度掉点硬数字"的关键，前提是先轮换 Kaggle/HF 凭据。

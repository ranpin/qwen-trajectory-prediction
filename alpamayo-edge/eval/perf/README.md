# 性能产物（跨设备加速比 · 引擎容量 · 精度选择交叉点）

| 文件 | 内容 |
|---|---|
| `fp16_speedup_goat.json` | 同机三方（FP16/INT8/INT4）TTFT/TPOT/功耗/能效，orin-goat 64GB。FP16 基线的唯一来源 |
| `len4096_sweep.csv` | **2026-07-27** 新增：`maxInputLen` 1024 vs 4096 两套引擎的 prefill 128–4096 与 decode 200 迭代实测（orin-dog） |
| `crossover_prediction.json` | 精度选择交叉点的**事前预测 + 事后对账**（含被证实/被修正的每一条） |
| `crossover_viz.py` | → `docs/figures/precision_crossover.png` |

## 为什么要建第二套引擎

`maxInputLen = 1024` 一直是 `llm_build` 的**默认值**，我们从未设过（见 PROBLEMS §H1/§H2）。
n=210 基准里最长的一题实测 1015 token，**只剩 9 个 token 余量**；而真实相机分辨率远高于我们缩到的
448 px——一帧原生 1080p 对齐到 32 的整数倍后是 1920×1088 = **2040 image token**，单帧就超过 1024。

因此新建 `engines/{int4,int8}_len4096`（`--maxInputLen 4096 --maxBatchSize 1 --maxKVCacheCapacity 8192`）
与 `engines/int4_len4096/visual`（`--maxImageTokens 4096 --maxImageTokensPerImage 2048`）。
**老引擎 `engines/{int4,int8}` 一律不动**——所有已发布数字出自它们，重建会破坏可比性。

### 三个上限的层级（真正卡分辨率的不是 maxInputLen）

| 上限 | 默认 | 管什么 |
|---|---|---|
| `maxImageTokensPerImage` | **512** | **单帧分辨率**（512 token = 0.52 MP ≈ 960×544）——真正的瓶颈 |
| `maxImageTokens` | 1024 | 所有帧合计；ViT `maxHW = 4 ×` 此值（即 PROBLEMS E3 的 4096 patch 上限） |
| `maxInputLen` | 1024 | 图 + 文合计进 LLM；TRT profile `opt = maxInputLen/2` |
| `max_pixels`（HF processor） | 2,097,152 | 绝对天花板 = 2048 token/图；车规 2880×1888 会被它缩掉 |

## 复现

```bash
# 设备端（orin-dog）：建引擎 + 扫描
bash scripts/build_len4096.sh      # 3 个引擎，记录 build 峰值内存
bash scripts/sweep_len4096.sh      # -> bench/len4096_sweep.csv
bash scripts/build_controls.sh     # 归因 INT8 的 opt 位移代价（两个单变量对照）
# 本地：出图
.venv/bin/python alpamayo-edge/eval/perf/crossover_viz.py
```

## 关键结论

- **容量代价几乎为零**：引擎文件大小不变（INT4 5.7 G / INT8 8.9 G / 视觉 1.1 G 在 1024 与 4096 下相同）——
  `maxInputLen` 只影响激活与 profile，不影响权重。
- **INT4 对 profile 形状完全不敏感**（≤0.5%）；**INT8 慢 15%**，两个单变量对照把它拆开了
  （`len4096_controls.csv`）：**KV 容量无影响**（−0.8%/−1.8%）、**maxBatchSize 4→1 占 +6.0%/+6.7%**
  （反直觉：缩小 profile 的最大 batch 反而让 batch-1 的 prefill 变慢）、**opt 形状 512→2048 占 +8.8%/+8.3%**。
  推测机制：INT4 走 W4A16（反量化后 fp16 GEMM，tactic 对形状不敏感），INT8 走原生 int8 张量核，
  tactic/tile 空间更大也更挑形状——**机制是推断，未证明**。
- **TTFT@4096（INT4）= 2554.6 ms**，事前预测 2650 ms，**误差 −3.6%** ⇒ 长输入 prefill 可由
  「matmul 2·P·n + attention 144·n²·d，除以实测 27.1 TFLOP/s」预测到 4% 内。
- **decode 200 迭代**：INT4 30.78 ms、INT8 49.47 ms。此前 10 迭代的 32.34/33.93 ms 偏高 4.8–9.3%，是噪声。
- **精度选择随负载形状翻转**：`INT8 更快 ⟺ 输出 token < 输入 token / 83`（同配置对；各自最优配置下为 /66）。
  n=210 基准（759 in / 1 out）**实测 INT8 快 1.43×**——与站点头条"decode 看 INT4"给人的印象相反，
  因为该负载是 prefill 主导，而权重量化对 prefill 无效（INT4 prefill ≈ FP16）。

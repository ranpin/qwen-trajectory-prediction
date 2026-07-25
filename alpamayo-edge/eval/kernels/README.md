# 手写 W4A16 GEMV 微基准（K2）

## 目的
Nsight（见 `eval/profile/`）显示 TRT-Edge-LLM 的 `gemv_kernel` 占 INT4 decode 时间 72.1%、
有效带宽 147.9 GB/s = 理论峰值 204.8 的 72.2%。但**「占理论峰值 72%」只有在硬件真能跑满时才代表有余量**，
所以本基准**先测这台机器的真实可达读带宽**，再拿手写 kernel 对照。

## 复现
```bash
scp w4a16_gemv.cu vision@<orin>:/home/vision/kbench.cu
ssh vision@<orin> 'export PATH=/usr/local/cuda/bin:$PATH && \
  nvcc -O3 -arch=sm_87 -o kbench kbench.cu && ./kbench > kbench_results.csv'
scp vision@<orin>:/home/vision/kbench_results.csv ./results.csv
.venv/bin/python eval/kernels/gemv_bench_viz.py     # -> docs/figures/gemv_kernel_bench.png
```
注意：`nvcc` 不在非交互式 ssh 的 PATH 里，需显式 `export PATH=/usr/local/cuda/bin:$PATH`。

## 结果（orin-dog，sm_87，MAXN）
| | 有效带宽 | 占理论峰值 | 占**可达**带宽 |
|---|---|---|---|
| 实测可达读带宽（1GB 流式读，128-bit load） | **152.3 GB/s** | 74.4% | 100% |
| TRT-Edge-LLM `gemv_kernel` | 147.9 GB/s | 72.2% | **97.1%** |
| 手写 v4（x 入 shared + half2） | 79.2 GB/s | 39% | 52.0% |
| 手写 v1（标量 32-bit load） | 56.9 GB/s | 28% | 37% |
| 手写 v2/v3（128-bit load，无 shared x） | ~20 GB/s | 10% | 13% |

**核心结论：TRT 的 GEMV 已达可达带宽的 97.1%，最多只剩 3.0% 余量（端到端约 +2%）⇒ 手写 kernel 这条路不通。**
上一轮基于「理论峰值」推出的「+10% 余量」预测**由此被推翻**——分母错了。

## 教训
1. **优化前先测天花板**。对着理论峰值算余量会得出错误结论；152.3 GB/s 这个数两次运行差 0.4%，
   且与 NVIDIA 论坛社区实测 151.7 GB/s（D2D）独立吻合。
2. **v2/v3 把 32-bit load 换成 128-bit 反而更慢**：瓶颈不是 DRAM，而是逐元素重复读 `x` 与标量
   int→fp16 转换。v4 把 x 放进 shared memory（block 内 8 warp 复用）+ half2 后才快 1.4–4×。
3. 与 TRT 仍差 1.87×，补齐需 AWQ/FasterTransformer 的 `lop3` 位技巧（直接拼 fp16 尾数、免转换指令）
   \+ 权重离线置换；但因 ①，补齐也没有端到端收益，故**主动停手**。
4. **Jetson 会按负载调 GPU/EMC 频率，`jetson_clocks` 需 root（不可用）** → 计时前先跑 400 次
   持续预热再取最优值；否则短脉冲 kernel 测出的带宽偏低约 30%（首版实测天花板只有 104.9 GB/s，是错的）。

## 诚实边界
- v4 用 half2 累加，相对误差 0.1–0.7%（v1 float 累加 0.04%）；产品化需改回 float 累加并再损失少量速度。
- 本基准是**独立 kernel**，未做成 TRT plugin 端到端集成（因结论是收益 ≈2%，不值得）。
- 权重为随机值、layout 为我自定义的 AWQ 风格（int4 packed + fp16 scale/zero, group 128），
  与 TRT 内部 layout 不必逐位相同；对照的是**带宽效率**这一口径。

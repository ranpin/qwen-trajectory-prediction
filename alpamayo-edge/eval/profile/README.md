# Kernel 级剖析产物（Nsight Systems，orin-dog）

## 复现命令（在 Orin 上）
```bash
source ~/.bashrc                      # 提供 EDGELLM_PLUGIN_PATH
cd /home/vision/TensorRT-Edge-LLM
nsys profile -o /home/vision/prof/int4_dec_node --force-overwrite true \
  --trace cuda --sample none --cpuctxsw none --cuda-graph-trace=node \
  ./build_orin/examples/llm/llm_bench --engineDir /home/vision/engines/int4/llm \
  --mode decode --pastKVLen 512 --iterations 20 --warmup 3
nsys stats --report cuda_gpu_kern_sum --format csv int4_dec_node.nsys-rep
nsys stats --report cuda_gpu_trace  --format csv int4_dec_node.nsys-rep   # grid/block 维度
```

**`--cuda-graph-trace=node` 是必须的**：TRTEdge 的 decode 走已捕获的 CUDA graph，
默认 `--cuda-graph-trace=graph` 会把 20 次计时迭代折叠成一个区间，
只有 3 次慢速 warmup 被逐 kernel 归因 → 份额与带宽都会算错
（实测：warmup 口径 gemv 73.4%/绝对时间偏高 ~40%；node 口径 72.1% 且四项合计 33.8 ms ≈ 实测 TPOT 33.1 ms）。

## 本地归档
- `int4_decode_kern_sum.csv` —— steady-state kernel 汇总（node 口径，24 次迭代）。
- `.nsys-rep` / `.sqlite` 原始文件留在设备 `/home/vision/prof/`（数百 MB，未拉回；可由上面命令重建）。

## 关键量（详见 docs/edge_deploy_status.md「Kernel 级剖析」节）
| kernel | 每 token 耗时 | 占比 | 有效带宽 |
|---|---|---|---|
| W4A16 GEMV（每层 7 次） | 24.39 ms | 72.1% | 147.9 GB/s（峰值 72.2%） |
| lm_head fp16 GEMM（N=151936） | 7.46 ms | 22.0% | 166.9 GB/s（81.5%） |
| attention `kernel_mha` | 0.91 ms | 2.7% | — |
| 已融合 elementwise | 1.08 ms | 3.2% | — |

- lm_head 身份由 grid 维度确认：`grid=(1,1583,1)`、tile 64×96 → 1583×96 = 151968 ≈ vocab 151936。
- **Nsight Compute (`ncu`) 不可用**：需 root，本机 `sudo` 需密码 → 硬件计数器（`dram__bytes_*`）拿不到；
  带宽口径改为「解析字节 ÷ 实测 kernel 耗时」，其字节数已与引擎文件大小自校验到 +0.13%。

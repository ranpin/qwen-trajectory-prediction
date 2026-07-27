# 设备端脚本（Orin 上实际跑的那几个）

从两台 Orin 归档到仓库的原始脚本。设备上的副本在各自的
`chenrunbin/alpamayo-edge/scripts/`（见 [`../../docs/device_layout.md`](../../docs/device_layout.md)）。

| 脚本 | 设备 | 作用 | 产出 |
|---|---|---|---|
| `measure_power.sh` | orin-dog (32GB) | bench 期间并行采 `tegrastats`，算**三轨之和**（VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0）的 GPU-active 均值与能效 | `logs/pw_*.log` → §1.4 功耗/能效 |
| `dog_fp16_validate.sh` | orin-dog | 把 goat 建好的 FP16 引擎（16.4 GB）拉回 32GB 机，试"**build 不下、但拿现成引擎能不能跑**" | `dog_fp16_validate.out`：传输完整、引擎 15.15 GB，但**推理未跑起来**（见下） |
| `build_bench.sh` | orin-goat (64GB) | 解包 ONNX → 建 INT4/INT8 引擎（与 FP16 同参数）→ 三精度 prefill/decode bench | `logs/build_int{4,8}.log` → §1.1 加速比 |
| `q_power.sh` | orin-goat | INT4/INT8 功耗（同 tegrastats 口径） | `logs/q_power.out` |
| `fp16_power.sh` | orin-goat | FP16 功耗，量化能效比的分母 | — |

## 注意
- **`dog_fp16_validate.out` 里 `FP16_PREFILL:` / `FP16_DECODE:` 都是空的，`PEAK_INFER_MEM_MB=1468`**
  ——1468 MB 只是这台机的空载内存，说明 `llm_bench` **根本没跑到分配 15 GB 那一步就退了**：
  FP16 引擎在 32GB 的 dog 上**反序列化即 OOM**（要 15 GB，而 16.4 GB 的 tar 文件自身页缓存已把 30 GB 吃满，
  无 sudo 密码无法 drop cache）。**这是一次失败的验证，不是成功的**。
  结论因此更硬：FP16 在 32GB 机上 **build（峰值 54.8 GB）与 load（15 GB + 页缓存 ≈ 30 GB）双双不可行**
  ⇒ 量化不只是提速，是**能否部署**的前提。详见 [`../../docs/orin_goat_migration_plan.md`](../../docs/orin_goat_migration_plan.md) Phase 4.5。
- 已发布的 FP16 基线（TTFT 298 ms / TPOT 89.3 ms）全部来自 **orin-goat（64GB）**，不是这台机。
- 脚本里的绝对路径已随 2026-07-27 的设备整理更新（原先写的是 `/home/vision/...` 顶层与
  `/home/nvidia/chenrunbin.crb/...`）；脚本头有注释标明改动。
- `measure_power.sh` 原先把 `pw_*.log` 写到**共享账号的家目录顶层**，这也是设备上文件散落的原因之一，
  现已改为写进项目自己的 `logs/`。

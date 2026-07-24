# orin-goat (64GB) 迁移与 FP16 补全计划

> **状态(2026-07-24):核心已完成** ✅ —— FP16 引擎在 goat build 成功(峰值 54.8GB)、测得性能+功耗;同机三方加速比 decode INT4 2.66×/INT8 1.70×、E2E 2.5×;跨设备 INT4/INT8 与 dog <3% 复现。结果见 `eval/perf/fp16_speedup_goat.json`。
> **未完成的一项(Phase 4.5)**:把 FP16 引擎拷回 dog 实跑——因 **goat 出站大文件传输被截断在 ~4.6GB**(Mac→goat 入站正常,goat 上行受限)未完成。"FP16 推理装得下 dog 30GB"以**内存账**为据(引擎 15.15GB + 实测激活 8MB + KV ~1.2GB ≈ 17GB),非 dog 实跑。goat 上完整引擎保留在 `/home/nvidia/chenrunbin.crb/`,后续可用同 LAN 直传补做。

**目标**:利用 64GB 的 orin-goat 建出 FP16 引擎（32GB 的 orin-dog 因 build 期 OOM 建不出），**补上缺失的 FP16 性能基线（量化 vs FP16 加速比）**，并做**跨设备复现**验证。

## 前置（需用户操作）
- **设备**:orin-goat = `nvidia@30.245.40.73`，AGX Orin **64GB**，同架构 sm_87。工作目录 **`/home/nvidia/chenrunbin.crb`**（用户已建）。
- **访问（待授权）**:把本机 RSA 公钥（`id_rsa`，能上 orin-dog 的那把）加到 goat 的 `/home/nvidia/.ssh/authorized_keys`。加完我即可自主执行以下 Phase。
- 本阶段只用**已有 ONNX 产物**（不重下模型/不重量化）→ 不涉及 HF 门控/凭据。

## 关键判断：引擎可移植性
TRT 引擎绑定 **GPU 架构 + TensorRT 版本**。两机同为 sm_87：
- **若 goat 的 TensorRT 版本 = dog 的 10.7.0.23** → 在 goat build 的 `.engine` **可拷回 dog 直接跑**（FP16 推理仅需 ~17GB，dog 的 30GB 够）。
- **若版本不同** → 引擎不通用，但可在 goat 本机 build+跑（64GB 足够 build）。

## Phase 0 — 侦察 goat（登上后第一步）
```bash
hostname; tr -d '\0' </proc/device-tree/model; free -g
cat /etc/nv_tegra_release | head -1              # L4T/JetPack
cat /usr/local/cuda/version.json | grep version # CUDA
dpkg -l | grep libnvinfer-bin                    # TensorRT 版本（决定引擎可移植性）
ls /home/vision/TensorRT-Edge-LLM/build_orin     # TRTEdge 是否已就绪
df -h /home
```

## Phase 1 — TRTEdge 就绪
- 若 goat 已有 `build_orin/` 且版本匹配：直接用。
- 若无：按**正确配方**构建（务必带 `-DEMBEDDED_TARGET=jetson-orin`，否则 sm_87 FMHA 被排除→崩，见 [orin_build_notes.md](orin_build_notes.md) / [PROBLEMS.md](PROBLEMS.md) B1）：
  ```bash
  cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=<goat 的 CUDA>
  make -j$(nproc)
  ```

## Phase 2 — 传产物到 goat（ONNX，走内网）
从 orin-dog 直接 scp（或从本地 `alpamayo-edge/outputs/`）：
```bash
# fp16(必需) + int4/int8(跨设备复现用) + 视觉塔 + 基准帧
scp vision@30.245.40.99:/home/vision/edge_fp16_llm.tgz .          # 12G
scp vision@30.245.40.99:/home/vision/edge_artifacts_int4_awq.tgz .
scp -r vision@30.245.40.99:/home/vision/edge_int8 .
scp -r vision@30.245.40.99:/home/vision/engines/int4/visual .
scp vision@30.245.40.99:/home/vision/bench_frames.tgz .           # 基准 210 帧+标注
```

## Phase 3 — 建引擎（goat，64GB，FP16 现在能建）
```bash
export EDGELLM_PLUGIN_PATH=<goat>/build_orin/libNvInfer_edgellm_plugin.so
llm_build --onnxDir Cosmos-8B-fp16-onnx-llm --engineDir engines/fp16/llm   # 关键：验证不再 OOM
# 同法建 int4/int8（跨设备复现）
```
> 监控 `free -m`：若 build 期用量峰值确实需 >30GB，64GB 应从容通过 → 直接印证"是 build 内存墙、非算法问题"。

## Phase 4 — 实验（补全 + 复现）
1. **FP16 性能基线（最重要，补缺口）**：`llm_bench` 测 FP16 的 prefill(TTFT)/decode(TPOT)，得 **量化 vs FP16 的加速比**（现在只有 INT4-vs-INT8）。
2. **FP16 显存/功耗**：tegrastats。
3. **跨设备复现**:在 goat 重跑 INT4/INT8 的 TTFT/TPOT/TPS，与 dog 对比（同 sm_87，应一致或近似）——增强可复现性。
4. （可选）**FP16 on-device 精度**:在 goat 跑 210 基准的 FP16 引擎，与云端 PyTorch FP16(88.2%/76.2%) 交叉核对。
5. （可选，若愿建 FP16 引擎拷回 dog）验证 goat→dog 引擎可移植 + dog 上 FP16 推理确实 ~17GB 跑得起来。

## Phase 5 — 回写
- 性能表补 **FP16 列** → 三精度完整（FP16/INT8/INT4 的 TTFT/TPOT/TPS/E2E + 加速比）。
- serving_metrics 图加 FP16 曲线；footprint 图加 FP16。
- 记 **跨设备复现** 结论 + goat 环境版本进 METHODOLOGY §0。
- 若引擎可移植成立：把"FP16 装不下"的边界更新为"dog 不能 build 但能跑；goat 能 build"。

## 风险
- TRT 版本不一致 → 引擎不可移植（退化为 goat 本机跑，不影响补 FP16 基线）。
- goat 未装 TRTEdge/JetPack 差异 → 需 Phase 1 重建（用我们的正确配方，规避 FMHA 坑）。
- 门控模型/凭据：本阶段用**已有 ONNX 产物**（不需重新下模型/量化），无需 HF 门控。

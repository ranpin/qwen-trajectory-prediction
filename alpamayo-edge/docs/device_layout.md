# 文件放在哪：各设备目录布局（2026-07-27 整理）

> 本项目横跨 **1 台 Mac + 2 台 Jetson Orin + Kaggle 云**，产物合计约 **230 GB**。
> 两台 Orin 都是**共享账号**（多人共用 `vision` / `nvidia` 登录），所以我的文件统一收在
> **`<home>/chenrunbin/alpamayo-edge/`** 下，各目录内附 `INDEX.md`。此前散落在家目录顶层，容易被误删/误认。

## 一、总览

| 位置 | 路径 | 大小 | 角色 | 索引 |
|---|---|---|---|---|
| **本地 Mac** | `~/projects/qwen-trajectory-prediction/`（git 仓库，分支 `alpamayo-edge`） | 27 G | 代码 / 文档 / 站点 / 归档 | 本仓库 `README.md` |
| **orin-dog** `vision@30.245.40.99` | `/home/vision/chenrunbin/alpamayo-edge/` | 98 G | **主部署机**（32GB 模组）：INT4/INT8/2B 建引擎 + 全部实测 | 机上 `INDEX.md` |
| **orin-goat** `nvidia@30.245.40.73` | `/home/nvidia/chenrunbin/alpamayo-edge/` | 102 G | **FP16 基线机**（64GB）：dog 上 build 不下的 FP16 在此建成 | 机上 `INDEX.md` |
| orin-elk `vision@30.245.40.123` | `/home/vision/chenrunbin/` | 12 K（空） | **无本项目文件** | — |
| pc-3070 `vision@30.245.40.164` | `/home/vision/chenrunbin/` | 104 K（`agentguide`，他项目） | **无本项目文件** | — |
| **scm** `chenrunbin.crb@100.82.100.41` | 待建 | **18 TB 空闲** | **归档盘**（2026-07-29 定位）：**不能下载** —— `huggingface.co` TCP 443 不可达（`pypi.org` 通 ⇒ 白名单受限出网）、且无 ffmpeg/ffprobe。用途 = 存本地"即下即删"产出的帧与标注长期副本 | — |
| **Kaggle 云** | 私有 kernel `alpamayo-edge-export` / `-2b` / `-lmhead` | — | 量化 + ONNX 导出 + FP16 参考；**输出即时下载，云端不留** | `cloud/README.md` |
| **GitHub** | `ranpin/qwen-trajectory-prediction` | — | `main`（旧项目）· `alpamayo-edge`（源）· `gh-pages`（[已发布站点](https://ranpin.github.io/qwen-trajectory-prediction/)） | — |

## 二、两台 Orin 的统一分类

同一套子目录名，两台一致：

| 子目录 | 内容 | dog | goat | 可重建？ |
|---|---|---|---|---|
| `engines/` | TensorRT 引擎（**sm_87 专用**，不可跨架构） | 28 G（int4/int8/int8_domain/2b） | 30 G（**fp16 16 G** + int4 + int8） | ✅ 由 `onnx/` 经 `llm_build` |
| `onnx/` | 云端量化导出的 ONNX（建引擎的输入） | 44 G | 32 G | ✅ 由 `tarballs/` 解包 |
| `tarballs/` | 传输/迁移打包 | 26 G | 40 G | ✅ **纯冗余，可删** |
| `bench/` | n=210 基准输入 / 输出 / 日志 / 1260 采样帧 | 35 M | — | ⚠️ 输出是实测记录 |
| `io/` | 各次推理的请求/响应 JSON | 152 K | — | ❌ 实测记录 |
| `logs/` | 构建 / 功耗 / 运行原始日志 | 1.3 M | 316 K | ❌ 实测记录 |
| `prof/` | Nsight Systems 剖析 | 6.6 M | — | ❌ 实测记录 |
| `kernels/` | 手写 W4A16 GEMV 微基准 | 1.1 M | — | ❌ 结果已入 git |
| `scripts/` | 设备端脚本（功耗采样、建引擎+bench） | 8 K | 16 K | ❌ 已入 git（`scripts/orin/`） |
| `trtedge/` | TRTEdge 已编译二进制 | 用共享安装 `/home/vision/TensorRT-Edge-LLM/`（**不在我目录**） | 92 M（从 dog 拷来） | ✅ |

### dog 上保留的两个兼容软链
`/home/vision/engines` → `chenrunbin/alpamayo-edge/engines`
`/home/vision/bench` → `chenrunbin/alpamayo-edge/bench`

原因：**已入 git 的运行记录里写死了旧绝对路径**（如
`eval/accuracy/benchmark/orin_int4_outputs_n210.json` 里的 `/home/vision/bench/frames/robovqa_0_0.jpg`、
`eval/vision/io/worst_req.json`）。这些是实测记录，改写就不再忠实，所以**留软链而不改记录**，
老命令与老记录都能原样重跑。确认无人依赖后可删软链。

### 不属于我、整理时未动的
- `/home/vision/TensorRT-Edge-LLM/`（**共享框架安装**，`~/.bashrc` 的 `EDGELLM_PLUGIN_PATH` 指向它；
  移动会让所有人的推理崩，见 [`PROBLEMS.md`](PROBLEMS.md) B2/F1）
- `/home/vision/llama.cpp`（2026-07-07 的纯净上游克隆，无构建产物、无模型 ⇒ **无法归属**，未动）
- 他人文件：`ywz/` `ywz.log` `zwx/` `inference_images/` `activevision.log` `carcontrol.log` `find.log` `NV_*.sh`

## 三、本地 Mac 的分类

```
alpamayo-edge/
├── cloud/        Kaggle / Colab 量化导出脚本（入 git）
├── docs/         METHODOLOGY / PROBLEMS / 本文 / figures/（入 git）
├── eval/         全部评测脚本 + 整理后的结果与原始 I/O（入 git）
│   ├── accuracy/ 精度：探针集 + n=210 基准 + 校准集消融
│   ├── vision/   视觉编码器扫描（含 io/ 逐字请求响应）
│   ├── kernels/  手写 GEMV 微基准源码 + 结果
│   ├── perf/     跨设备加速比
│   └── profile/  Nsight 汇总 CSV
├── scripts/orin/ 设备端脚本归档（入 git）
├── site/         发布站点源（入 git → gh-pages）
└── outputs/      **gitignore**，仅本地：
    ├── *.tgz             自产量化权重 4 件 26.7 G + bench_frames.tgz 30 M
    └── archive/          原始日志（orin_logs/ goat_logs/ orin_prof/）+ MANIFEST.md（SHA256）
```

**入 git 与只在本地的分界**：小而经过整理、被文档引用的 → git；原始大件与原始日志 → `outputs/`（本地 + MANIFEST）。
设备专用可重建大件（TRT 引擎）→ 只在设备上，不拉回本地（见 [`../outputs/archive/MANIFEST.md`](../outputs/archive/MANIFEST.md)）。

## 四、单点风险与磁盘

| 东西 | 只存在于 | 丢了怎么办 |
|---|---|---|
| **FP16 引擎** 16 G | **仅 orin-goat** `engines/fp16` | ✅ 可重建：ONNX 已在本地归档（`edge_fp16_llm.tgz`，SHA256 已校验），但**必须在 ≥64 GB 内存的机器上** build（32GB 机峰值 54.8 G 会 OOM） |
| INT4/INT8/2B 引擎 | 两台 Orin | ✅ 由本地 ONNX 重建（sm_87 机器即可） |
| 全部原始日志 / 请求响应 / 剖析 | 已于 2026-07-27 **全部拉回本地** | ✅ 本地 + MANIFEST |
| 自产量化 ONNX | 本地 + 设备 | ✅ 本地有 4 个 tgz 且 SHA256 已校验；最坏情况重跑 Kaggle 量化 |

**磁盘**：
- orin-goat `/home` 已用 **91%**（剩 40 G）⇒ 要腾空间先删 `tarballs/`（40 G，与已解包内容完全重复）。
- orin-dog `/home` 用 41%（NVMe 916 G，剩 521 G），宽松；`tarballs/` 亦可删回收 26 G。
- 两台的 `tarballs/` 都是**纯冗余**（内容 = `onnx/` + goat 的 `engines/fp16`），删前对照本地 MANIFEST 的 SHA256 即可。

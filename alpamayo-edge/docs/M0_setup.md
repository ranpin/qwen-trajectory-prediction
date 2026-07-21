# M0 — 环境搭建（命令清单）

两处环境:**导出/量化主机(x86+GPU 或免费云)** 与 **Orin(边缘 build+run)**。
命令取自 NVIDIA/TensorRT-Edge-LLM 官方 installation.md(v0.9.0)。

---

## A. 导出/量化主机(x86 + NVIDIA GPU;8GB 3070 装不下 8B/10B → 用免费云,见 `cloud/`)
```bash
# 推荐容器(干净环境):
docker pull nvcr.io/nvidia/pytorch:25.12-py3
docker run --gpus all -it --rm -v $PWD:/w nvcr.io/nvidia/pytorch:25.12-py3

git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
pip3 install ".[tools]"                 # 装 tensorrt-edgellm-* 入口(含量化工具)
pip3 install -r requirements-server.txt # 可选:实验性 server
export EDGE_LLM_PATH=$PWD && export PYTHONPATH=$EDGE_LLM_PATH:$PYTHONPATH
tensorrt-edgellm-export --help          # 自检
tensorrt-edgellm-quantize --help
hf auth login                           # 下载 gated 权重需要
```
> ⚠️ 导出 10B 内存开销大(FP8 导出官方注明可达模型 20× CPU 内存;我们用 FP16/INT4/INT8 略低,
> 但 10B 仍需较大 CPU RAM)。免费 Kaggle(2×T4=32GB GPU + ~30GB RAM)一次性跑,见 `cloud/kaggle_export_quant.sh`。

## B. Orin(aarch64,JetPack)—— build engine + 推理
平台矩阵(installation.md):
| 平台 | JetPack | CUDA_CTK_VERSION | 精度 |
|---|---|---|---|
| Jetson Orin | 6.2+ | 12.6 | FP16 / INT8 / INT4 |
| Jetson Orin | 7.2 | 13.2 | FP16 / INT8 / INT4 |

```bash
# 确认 JetPack/CUDA
nvcc --version                          # 应匹配上表 CUDA_CTK_VERSION
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \       # ⚠️ 必须！否则默认 arch=80;86;89、-DEXCLUDE_SM_87 排除 sm_87 FMHA kernel → 推理必崩
  -DCUDA_CTK_VERSION=12.6                # 依你的 JetPack 改 12.6 或 13.2
cmake --build . -j
# 产物:build/examples/llm/llm_build, build/examples/multimodal/{visual_build,action_build,action_inference}
```
> ⚠️ **务必带 `-DEMBEDDED_TARGET=jetson-orin`**（本项目最大的坑）。漏掉它 CMake 会走默认分支、静默排除
> sm_87 的 FMHA kernel，引擎能建成但推理一进 attention 就 `There must be one kernel to implement the MHA` 崩。
> 配置期确认日志 `FMHA Kernels: Excluding SM architectures: ...` **只排除非 87 的架构**。完整根因见
> [orin_build_notes.md](orin_build_notes.md)。

## 自检清单
- [ ] 主机:`tensorrt-edgellm-export --help` / `--quantize --help` 正常
- [ ] 主机:`hf auth login` 完成,能 `hf download nvidia/Alpamayo-R1-10B`(许可已同意)
- [ ] Orin:`build/examples/...` 可执行文件存在
- [ ] Orin:`nvcc` 版本匹配 JetPack

## 下一步
- 主机(或免费云):`cloud/kaggle_export_quant.sh` 一次性产出
  `Alpamayo-R1-10B/onnx/{llm,visual,action}`(FP16)与 `Cosmos-Reason2-8B-int4/int8` 检查点+ONNX。
- scp 到 Orin → `scripts/export_and_build.sh` Step3 / `scripts/quantize.sh` 后半段的 build → 评测/基准。

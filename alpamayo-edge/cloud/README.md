# 一次性导出/量化（免费云）

8GB 3070 装不下 8B/10B,这一步在免费云一次性完成,产物(ONNX,几 GB)下载后 scp 到 Orin。

> ✅ **实际采用并跑通的脚本**：[`kaggle_cosmos_only.py`](kaggle_cosmos_only.py)（Kaggle 2×T4，headless，Cosmos-Reason2-8B
> INT4/INT8，含对 TRTEdge `quantize.py` 的三处补丁）。用 `kaggle kernels push` 提交即后台跑，产出 `edge_artifacts_*.tgz`。
> 下面的 `kaggle_export_quant.*` / `modelscope_*` / `colab_*` 是**早期含 Alpamayo 的多平台备选**，Alpamayo 量化不被支持后收敛到 Cosmos-only（见 [../docs/FINDINGS.md](../docs/FINDINGS.md)）。HF token 走**环境变量**，勿硬编码。

## 平台选择
- **Kaggle(当前采用,直连 GitHub/HF、免镜像)** → `kaggle_export_quant.ipynb`(交互跑)
  或 `kaggle_run.sh`(给我 Kaggle API token 后可headless驱动)
- ModelScope 魔搭(国内直连但 GitHub 需代理、HF 需镜像)→ `modelscope_export_quant.sh`
- 百度 AI Studio(免费 V100 32GB)→ 用 `modelscope_export_quant.sh` 同款
- Colab(多地区被封)→ `colab_export_quant.ipynb`

## Kaggle 步骤(交互)
1. New Notebook → 右侧 Settings:**Accelerator=GPU T4 x2**、**Internet=On**;
   Add-ons→Secrets 新增 `HF_TOKEN`(你的 HF read token;先在 HF 网页同意 Alpamayo-R1/Cosmos-Reason2 许可)。
2. 打开 `kaggle_export_quant.ipynb`(File→Import Notebook 或复制 cell),Run All。
3. 右侧 Output 面板下载 `edge_artifacts.tgz`。
> T4 单卡 16GB:Cosmos-8B 量化(offload)可行;Alpamayo-10B FP16 导出可能 OOM,脚本已排 Cosmos 在前,OOM 就先只交 Cosmos。

## ModelScope 步骤
1. 开一个**带 GPU 的免费 Notebook**(挑显存最大的免费实例)。
2. 新建 cell,依次:
   ```python
   !pip -q install -U huggingface_hub
   import os; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
   !huggingface-cli login          # 贴 HF token(先在 HF 网站同意 Alpamayo/Cosmos 许可)
   ```
3. 上传本目录的 `modelscope_export_quant.sh`,再一个 cell:`!bash modelscope_export_quant.sh`
4. 跑完在文件浏览器下载 `/mnt/workspace/edge_artifacts.tgz`。

## 交给 Orin
把 `edge_artifacts.tgz` scp 到 Orin(或发我,我用现有 SSH 处理):
```
scp edge_artifacts.tgz vision@30.245.40.99:~/tensorrt-edgellm-workspace/
# 在 Orin: tar xzf ~/tensorrt-edgellm-workspace/edge_artifacts.tgz -C ~/tensorrt-edgellm-workspace/
```
之后 M2–M4(build engine / 推理 / 评测 / 基准)我接管。

## ⚠️ 两个真实风险
- **免费 GPU 显存**:Cosmos-8B 量化 ~24GB 可行;**Alpamayo-10B FP16 导出**吃紧,免费实例可能 OOM
  → 先跑 Cosmos(脚本已排在前),Alpamayo 挑最大免费 GPU;实在不行只交量化轨道。
- **HF 门控访问**:Alpamayo/PhysicalAI-AV 是门控仓库;国内用 `HF_ENDPOINT=hf-mirror.com` + token 一般可下,
  若镜像不支持门控,需换直连 HF 的网络。这是国内跑这步唯一的真不确定点。

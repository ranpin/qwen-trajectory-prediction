# 一次性导出/量化（免费云）

8GB 3070 装不下 8B/10B,这一步在免费云一次性完成,产物(ONNX,几 GB)下载后 scp 到 Orin。

## 平台选择
- **ModelScope 魔搭(国内首选,直连)** → `modelscope_export_quant.sh`
- 百度 AI Studio(免费 V100 32GB)→ 用 `modelscope_export_quant.sh` 同款(改 WORK 路径即可)
- Colab(部分地区被封,慎用)/ Kaggle → `colab_export_quant.ipynb`

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

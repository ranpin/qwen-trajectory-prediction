# 视觉编码器扫描 + 最坏请求的原始请求/响应（实测记录）

从 orin-dog 归档的**逐字原始 I/O**，是 §1.7（视觉编码器代价）与
[`PROBLEMS.md`](../../../docs/PROBLEMS.md) §H1（输入长度余量只剩 9 token）两处结论的证据。
汇总数值见 [`../RESULTS.json`](../RESULTS.json)。

| 请求 | 响应 | 帧数 / 帧形状 | image token | 用途 |
|---|---|---|---|---|
| `vt_1.json` | `vt_out_1.json` | 1 帧 · 448×252 | 112 | 帧数扫描（视觉时间随 patch 数线性） |
| `vt_2.json` | `vt_out_2.json` | 2 帧 · 448×252 | 224 | 同上 |
| `vt_4.json` | `vt_out_4.json` | 4 帧 · 448×252 | 448 | 同上 |
| `vt_6.json` | `vt_out_6.json` | 6 帧 · 448×252 | 672 | 同上（基准评测用的就是 6 帧） |
| `vt_6.json` | `vt_out_6b.json` | 6 帧 · 448×252 | 672 | **同会话**重跑，作 5:4 对比的同会话基线（避免跨会话方差） |
| `vt_tall6.json` | `vt_tall6_out.json` | 6 帧 · **448×358** | **924** | 长宽比不同 ⇒ 每帧 154 token，推翻"每帧 112 token" |
| `worst_req.json` | `worst_out.json` | 6 帧 · 448×358 + 316 字符题干 | 924 | n=210 中**最长的一题**（`request_idx 94`）：`Computed Tokens = 1015` vs 引擎 `maxInputLen = 1024` ⇒ **余量 9 token** |

## 怎么复现
```bash
# 在 orin-dog（路径见 docs/device_layout.md）
A=/home/vision/chenrunbin/alpamayo-edge
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_inference \
  --engineDir $A/engines/int4/llm --multimodalEngineDir $A/engines/int4/visual \
  --inputFile $A/io/worst_req.json --dumpProfile
# 日志里读 "Total Image Tokens" 与 "Computed Tokens"
```

## 关于文件里的绝对路径
请求 JSON 里的 `"image": "/home/vision/bench/frames/robovqa_94_0.jpg"` 是**当时运行的原样记录，未改写**
——改了就不再是实测记录。2026-07-27 设备整理后 `/home/vision/bench` 保留为**软链**指向
`/home/vision/chenrunbin/alpamayo-edge/bench`，所以这些文件**原样仍可直接重跑**。

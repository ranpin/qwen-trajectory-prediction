# 驾驶域真实帧（闸门 2 实测产物，2026-07-29）

来源：`dgural/PhysicalAI-Autonomous-Vehicles-Sample`（公开，无需 gate）
clip `01d3588e-bca7-4a18-8e74-c6cfe9e996db.camera_front_wide_120fov.mp4`
—— NVIDIA PhysicalAI-AV 的多相机行车视频，与 Alpamayo / Cosmos 同一体系。

**原生规格实测**：`1920×1080 @ 30 fps, HEVC, 20.17 s / 605 帧, 13 MB`

按 §0.0.1 的预处理链均匀采 6 帧（`ffmpeg -vf fps=6/20.166667`），**原生分辨率未缩放**。

## 这批帧把此前的解析预测逐项验证了

| 项 | 实测 | 此前解析预测 |
|---|---|---|
| 原生对齐（32 的整数倍） | 1920×**1088** | 1920×1088 ✓ |
| **token/帧** | **2040** | **2040** ✓ |
| 6 帧合计 | 12,240 | 12,240 ✓ |
| 超 `maxInputLen=4096` | **3.0 倍** | "超 3 倍" ✓ |
| `engines/*_len4096` 实装 | **1 帧**原生 1080p | 1 帧 ✓ |
| 缩到长边 448（现基准口径） | 448×252 → 448×256 → **112 token/帧** | 112 ✓（与 n=210 基准里 1020 段一致） |

**原生 / 缩放的 token 比 = 18.2×** —— 这是我们一直在付的"降采样折扣"的确切数字。

⇒ **站点 §1.10 延迟预算表里"1 帧原生 1080p"与"6 相机 1080p"两行，从解析外推升级为实测支撑。**

## 工作流：边下边处理再删（本地峰值 13 MB）

```bash
F=01d3588e-bca7-4a18-8e74-c6cfe9e996db.camera_front_wide_120fov.mp4
curl -sL --http1.1 -o clip.mp4 \
  "https://huggingface.co/datasets/dgural/PhysicalAI-Autonomous-Vehicles-Sample/resolve/main/data/$F"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of default=nw=1 clip.mp4
ffmpeg -v error -i clip.mp4 -vf "fps=6/20.166667" -frames:v 6 -q:v 2 native_%d.jpg -y
rm clip.mp4      # 即下即删：13 MB -> 只留 6 帧共 1.25 MB
```

批量拉 10 个 clip 也只是 130 MB 峰值 / 12 MB 留存。

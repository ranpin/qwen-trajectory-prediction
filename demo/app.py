#!/usr/bin/env python3
"""Gradio demo for Qwen3-4B trajectory prediction.

Sends the SAME prompt format the model was trained on (via the shared
prompt_format module), overlays a constant-velocity baseline and (for preset
examples) the ground truth, and is aware of coordinate normalization so it can
drive either the world-frame model or a normalized-frame model.
"""

import os
import sys
import json
import time
import argparse
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gradio as gr

# Use a CJK-capable font for plot text when one is installed (avoids tofu boxes).
_available = {f.name for f in fm.fontManager.ttflist}
for _name in ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
              "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP"]:
    if _name in _available:
        plt.rcParams["font.sans-serif"] = [_name]
        break
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "common"))
from prompt_format import (build_user_prompt, SYSTEM_PROMPT,  # noqa: E402
                           FRAME_INTERVAL, OBS_LENGTH, PRED_LENGTH, direction_label)
from trajectory_norm import compute_transform, apply_transform, invert_transform  # noqa: E402

PRED_MARKER = "预测轨迹"
PRESET_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                           "eth_ucy_test_sample.jsonl")


# ----------------------------- parsing / physics -----------------------------
def parse_coords(text, only_after_marker=False):
    """Extract (x, y) coords from text; optionally only after the pred marker."""
    if only_after_marker and PRED_MARKER in text:
        text = text.split(PRED_MARKER, 1)[1]
    coords = []
    for line in text.split("\n"):
        if "(" in line and ")" in line:
            try:
                s, e = line.index("("), line.index(")")
                x, y = map(float, line[s + 1:e].split(","))
                coords.append([x, y])
            except (ValueError, IndexError):
                continue
    return np.array(coords) if coords else None


def cvm_predict(obs, n_pred=PRED_LENGTH):
    """Constant-velocity (mean over observation) extrapolation."""
    vel = (obs[-1] - obs[0]) / (len(obs) - 1)
    return np.array([obs[-1] + vel * (k + 1) for k in range(n_pred)])


def speed_and_direction(obs):
    avg_speed = round(float(np.mean(np.linalg.norm(np.diff(obs, axis=0), axis=1))
                            / FRAME_INTERVAL), 2)
    heading = np.arctan2(obs[-1, 1] - obs[0, 1], obs[-1, 0] - obs[0, 0])
    return avg_speed, direction_label(heading)


def ade_fde(pred, gt):
    n = min(len(pred), len(gt))
    if n < 1:
        return None, None
    pred, gt = pred[:n], gt[:n]
    ade = float(np.mean(np.linalg.norm(pred - gt, axis=1)))
    fde = float(np.linalg.norm(pred[-1] - gt[-1]))
    return ade, fde


# --------------------------------- presets -----------------------------------
def load_presets():
    """Return {label: {scene, history_text, gt}} from the real test sample."""
    presets = {}
    if not os.path.exists(PRESET_FILE):
        return presets
    with open(PRESET_FILE) as f:
        for i, line in enumerate(f):
            if i >= 12:
                break
            item = json.loads(line)
            user = next(m["content"] for m in item["messages"] if m["role"] == "user")
            gt_text = item["messages"][-1]["content"]
            scene = user.split("场景：", 1)[1].split("\n", 1)[0]
            hist_lines = [l for l in user.split("\n") if l.strip().startswith("t=")]
            presets[f"#{i+1} {scene}"] = {
                "scene": scene,
                "history_text": "\n".join(hist_lines),
                "gt": parse_coords(gt_text, only_after_marker=True),
            }
    return presets


# --------------------------------- plotting ----------------------------------
def plot_trajectory(obs, llm, cv, gt, scene):
    fig, ax = plt.subplots(figsize=(9, 8))
    if obs is not None:
        ax.plot(obs[:, 0], obs[:, 1], "b-o", lw=2, ms=6, label="Observed", alpha=0.9)
    # connect last observed point to each future for readability
    for series, style, lab in [(llm, "r--^", "LLM"), (cv, "g:s", "CVM baseline"),
                               (gt, "k-.", "Ground truth")]:
        if series is not None and len(series):
            if obs is not None:
                ax.plot([obs[-1, 0], series[0, 0]], [obs[-1, 1], series[0, 1]],
                        style[0] + ":", lw=1, alpha=0.4)
            ax.plot(series[:, 0], series[:, 1], style, lw=2, ms=6, label=lab, alpha=0.9)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"Trajectory prediction — {scene}")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    return fig


# ------------------------------- prediction ----------------------------------
def predict(scene, history_text, preset_label, normalize, api_url, model, presets):
    obs_world = parse_coords(history_text)
    if obs_world is None or len(obs_world) < 2:
        return plt.figure(), "输入的历史轨迹无法解析（至少需要2个点）。"

    gt_world = presets.get(preset_label, {}).get("gt") if preset_label else None

    # Transform into the frame the deployed model expects.
    origin, angle = compute_transform(obs_world, normalize)
    obs_feed = apply_transform(obs_world, origin, angle)
    avg_speed, dir_label = speed_and_direction(obs_feed)
    prompt = build_user_prompt(obs_feed, scene, avg_speed, dir_label)

    t0 = time.time()
    try:
        resp = requests.post(
            f"{api_url}/v1/chat/completions",
            json={"model": model,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": prompt}],
                  "max_tokens": 512, "temperature": 0.1},
            timeout=60)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return plt.figure(), f"调用 API 失败：{e}"
    latency = time.time() - t0

    pred_feed = parse_coords(text, only_after_marker=True)
    llm_world = (invert_transform(pred_feed, origin, angle)
                 if pred_feed is not None else None)
    cv_world = cvm_predict(obs_world)

    fig = plot_trajectory(obs_world, llm_world, cv_world, gt_world, scene)

    lines = [f"⏱ 推理延迟：{latency:.2f}s   |   归一化：{normalize}", ""]
    if gt_world is not None:
        la, lf = ade_fde(llm_world, gt_world) if llm_world is not None else (None, None)
        ca, cf = ade_fde(cv_world, gt_world)
        lines.append("误差对比（相对真值）：")
        if la is not None:
            lines.append(f"  LLM  ADE {la:.2f}m  FDE {lf:.2f}m")
        lines.append(f"  CVM  ADE {ca:.2f}m  FDE {cf:.2f}m")
        lines.append("")
    lines.append("模型输出：")
    lines.append(text)
    return fig, "\n".join(lines)


# --------------------------------- UI ----------------------------------------
def create_demo(api_url, model, normalize):
    presets = load_presets()
    default_label = next(iter(presets), None)
    default = presets.get(default_label, {}) if default_label else {}
    example_history = default.get("history_text",
                                 "\n".join(f"t={i*0.4:.1f}s: ({2.3+0.2*i:.2f}, {4.5+0.3*i:.2f})"
                                           for i in range(OBS_LENGTH)))

    with gr.Blocks(title="Qwen3-4B 轨迹预测系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚶 基于 Qwen3-4B 的行人轨迹预测")
        gr.Markdown("### 微调 → 量化 → Orin 部署 全链路 Demo（含 CVM 基线对比）")

        with gr.Row():
            with gr.Column(scale=1):
                preset_dd = gr.Dropdown(
                    choices=list(presets.keys()), value=default_label,
                    label="预设样例（真实 ETH/UCY，含真值）"
                    if presets else "预设样例（无：未找到测试集文件）")
                scene_input = gr.Textbox(label="场景描述",
                                         value=default.get("scene", "人行横道"))
                history_input = gr.Textbox(
                    label="历史轨迹（每行：t=Xs: (x, y)，共8点）",
                    lines=10, value=example_history)
                predict_btn = gr.Button("预测", variant="primary", size="lg")
            with gr.Column(scale=1):
                plot_output = gr.Plot(label="轨迹可视化（BEV）")
                text_output = gr.Textbox(label="输出 / 延迟 / 误差", lines=16)

        def on_preset(label):
            p = presets.get(label, {})
            return p.get("scene", ""), p.get("history_text", "")
        preset_dd.change(on_preset, inputs=[preset_dd],
                         outputs=[scene_input, history_input])

        predict_btn.click(
            lambda scene, hist, label: predict(scene, hist, label, normalize,
                                               api_url, model, presets),
            inputs=[scene_input, history_input, preset_dd],
            outputs=[plot_output, text_output])

        gr.Markdown("---")
        gr.Markdown(f"**API**: `{api_url}`  |  **model**: `{model}`  "
                    f"|  **normalize**: `{normalize}`")
        gr.Markdown("蓝=观测  红=LLM预测  绿=CVM基线  黑=真值")

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str,
                        default=os.environ.get("ORIN_API_URL", "http://localhost:8080"),
                        help="llama.cpp server URL (defaults to $ORIN_API_URL)")
    parser.add_argument("--model", type=str,
                        default=os.environ.get("ORIN_MODEL", "qwen3-4b-q4_k_m.gguf"))
    parser.add_argument("--normalize", choices=["none", "translate", "translate_rotate"],
                        default=os.environ.get("TRAJ_NORMALIZE", "none"),
                        help="must match the deployed model's training normalization")
    parser.add_argument("--server", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_demo(args.api_url, args.model, args.normalize)
    demo.launch(server_name=args.server, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

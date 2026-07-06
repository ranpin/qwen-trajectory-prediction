#!/usr/bin/env python3
"""Gradio demo for Qwen3-4B trajectory prediction."""

import os
import json
import argparse
import requests
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

def predict_trajectory(scene_desc, history_traj, pred_horizon, api_url):
    """Call llama.cpp server for trajectory prediction."""
    prompt = f"场景：{scene_desc}\n历史轨迹：{history_traj}\n请预测未来{pred_horizon}秒的轨迹。"

    try:
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            json={
                "model": "qwen3-4b-trajectory",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.1,
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        result = f"Error calling API: {e}"

    # Parse and visualize
    try:
        history_coords = parse_coords(history_traj)
        pred_coords = parse_coords(result)

        fig = plot_trajectory(history_coords, pred_coords, scene_desc)
    except Exception as e:
        fig = create_error_plot(f"Visualization error: {e}")

    return fig, result


def parse_coords(text):
    """Extract (x, y) coordinates from text."""
    coords = []
    for line in text.split('\n'):
        if '(' in line and ')' in line:
            try:
                start = line.index('(')
                end = line.index(')')
                xy_str = line[start+1:end]
                x, y = map(float, xy_str.split(','))
                coords.append([x, y])
            except (ValueError, IndexError):
                continue
    return np.array(coords) if coords else None


def plot_trajectory(history, pred, scene_desc):
    """Create trajectory visualization plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if history is not None and len(history) > 0:
        ax.plot(history[:, 0], history[:, 1], 'b-o', linewidth=2,
                markersize=6, label='历史轨迹', alpha=0.8)
        ax.annotate('起点', (history[0, 0], history[0, 1]),
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=10, color='blue')

    if pred is not None and len(pred) > 0:
        ax.plot(pred[:, 0], pred[:, 1], 'r--^', linewidth=2,
                markersize=8, label='预测轨迹', alpha=0.8)
        ax.annotate('预测终点', (pred[-1, 0], pred[-1, 1]),
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=10, color='red')

        # Connect last history point to first prediction
        if history is not None and len(history) > 0:
            ax.plot([history[-1, 0], pred[0, 0]],
                   [history[-1, 1], pred[0, 1]],
                   'g:', linewidth=1, alpha=0.5)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'轨迹预测 - {scene_desc}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def create_error_plot(error_msg):
    """Create error message plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.5, error_msg, ha='center', va='center',
            fontsize=14, color='red', wrap=True)
    ax.axis('off')
    return fig


def create_demo(api_url="http://localhost:8080"):
    """Create Gradio demo interface."""

    example_history = """t=0.0s: (2.3, 4.5)
t=0.4s: (2.5, 4.8)
t=0.8s: (2.7, 5.1)
t=1.2s: (2.9, 5.4)
t=1.6s: (3.1, 5.7)
t=2.0s: (3.3, 6.0)
t=2.4s: (3.5, 6.3)
t=2.8s: (3.7, 6.6)"""

    with gr.Blocks(title="Qwen3-4B 轨迹预测系统") as demo:
        gr.Markdown("# 🚗 基于Qwen3-4B的轨迹预测系统")
        gr.Markdown("### 微调 → 量化 → Orin部署 全链路Demo")

        with gr.Row():
            with gr.Column(scale=1):
                scene_input = gr.Textbox(
                    label="场景描述",
                    placeholder="例如：十字路口，行人横穿马路",
                    value="人行横道"
                )

                history_input = gr.Textbox(
                    label="历史轨迹 (每行一个点: t=Xs: (x, y))",
                    lines=10,
                    value=example_history
                )

                horizon_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=0.5,
                    label="预测时长（秒）"
                )

                predict_btn = gr.Button("预测", variant="primary", size="lg")

            with gr.Column(scale=1):
                plot_output = gr.Plot(label="轨迹可视化")

                text_output = gr.Textbox(
                    label="模型输出",
                    lines=15
                )

        predict_btn.click(
            lambda scene, history, horizon: predict_trajectory(scene, history, horizon, api_url),
            inputs=[scene_input, history_input, horizon_slider],
            outputs=[plot_output, text_output]
        )

        gr.Markdown("---")
        gr.Markdown(f"**API Endpoint**: `{api_url}`")
        gr.Markdown("**模型**: Qwen3-4B (AWQ 4-bit quantized)")
        gr.Markdown("**部署**: NVIDIA Jetson AGX Orin + llama.cpp")

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str,
                        default=os.environ.get("ORIN_API_URL", "http://localhost:8080"),
                        help="llama.cpp server URL (defaults to $ORIN_API_URL)")
    parser.add_argument("--server", type=str, default="0.0.0.0",
                        help="Gradio server address")
    parser.add_argument("--port", type=int, default=7860,
                        help="Gradio server port")
    parser.add_argument("--share", action="store_true",
                        help="Create public share link")
    args = parser.parse_args()

    demo = create_demo(args.api_url)
    demo.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()

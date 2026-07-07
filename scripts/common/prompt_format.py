"""Canonical prompt/format helpers shared by data prep, eval, and the demo.

The training data, the deployed model, and the demo must all agree on the exact
user-prompt wording. Keeping the builder here (imported everywhere) prevents the
drift that previously made the Gradio demo send a different prompt than the model
was trained on.
"""

import math

FRAME_INTERVAL = 0.4   # seconds per frame
OBS_LENGTH = 8         # observed frames (3.2s)
PRED_LENGTH = 12       # predicted frames (4.8s)

SYSTEM_PROMPT = (
    "你是一个专业的行人轨迹预测专家。根据给定的行人历史轨迹和周围环境信息，"
    "预测该行人未来的运动轨迹。你需要先分析运动趋势，再给出预测坐标。"
)


def direction_label(angle_rad):
    """8-way Chinese compass label for a heading in radians."""
    deg = math.degrees(angle_rad) % 360
    if 337.5 <= deg or deg < 22.5:
        return "东"
    elif deg < 67.5:
        return "东北"
    elif deg < 112.5:
        return "北"
    elif deg < 157.5:
        return "西北"
    elif deg < 202.5:
        return "西"
    elif deg < 247.5:
        return "西南"
    elif deg < 292.5:
        return "南"
    else:
        return "东南"


def format_coords(coords, time_offset=0.0, frame_interval=FRAME_INTERVAL):
    """Render a list of (x, y) as 't=Xs: (x, y)' lines."""
    lines = []
    for i, (x, y) in enumerate(coords):
        t = time_offset + i * frame_interval
        lines.append(f"t={t:.1f}s: ({x:.2f}, {y:.2f})")
    return "\n".join(lines)


def build_user_prompt(obs_coords, scene, avg_speed, dir_label,
                      obs_len=OBS_LENGTH, pred_len=PRED_LENGTH,
                      frame_interval=FRAME_INTERVAL):
    """Build the exact user message the model was trained on.

    `scene` is the full scene string (callers add any suffix like
    '（行人密集区域）' themselves). `obs_coords` is a list/array of (x, y).
    """
    obs_text = format_coords(obs_coords, frame_interval=frame_interval)
    return (
        f"场景：{scene}\n"
        f"行人历史轨迹（过去{obs_len * frame_interval:.1f}秒，每{frame_interval}秒采样）：\n"
        f"{obs_text}\n"
        f"平均速度：{avg_speed}m/s，主要方向：{dir_label}\n\n"
        f"请预测该行人未来{pred_len * frame_interval:.1f}秒的轨迹。"
    )

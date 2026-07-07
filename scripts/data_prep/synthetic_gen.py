"""Generate synthetic pedestrian trajectory data for LLM fine-tuning.

Generates diverse trajectories using simple motion models:
- Linear (constant velocity)
- Acceleration/deceleration
- Curved paths (turning)
- Random walk with drift

Each sample is output in ms-swift chat format.
"""

import os
import sys
import json
import math
import argparse
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from trajectory_norm import normalize_obs_pred, MODES  # noqa: E402
from prompt_format import build_user_prompt, system_prompt  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic"
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

SYSTEM_PROMPT = (
    "你是一个专业的行人轨迹预测专家。根据给定的行人历史轨迹和周围环境信息，"
    "预测该行人未来的运动轨迹。你需要先分析运动趋势，再给出预测坐标。"
)

SCENE_TYPES = [
    "人行横道", "商业街", "校园广场", "公园步道",
    "地铁站出口", "住宅区道路", "商场走廊", "体育场馆",
]

VEHICLE_SCENE_TYPES = [
    "高速公路", "城市快速路", "十字路口", "环岛",
    "高速匝道", "主干道", "停车场", "隧道",
]

OBS_LENGTH = 8
PRED_LENGTH = 12
FRAME_INTERVAL = 0.4


def generate_linear(n_obs, n_pred, speed, direction, noise_std=0.02):
    """Constant velocity trajectory with slight noise."""
    total = n_obs + n_pred
    vx = speed * math.cos(direction)
    vy = speed * math.sin(direction)

    points = []
    x, y = random.uniform(-2, 2), random.uniform(-2, 2)
    for _ in range(total):
        points.append((x + np.random.normal(0, noise_std),
                        y + np.random.normal(0, noise_std)))
        x += vx * FRAME_INTERVAL
        y += vy * FRAME_INTERVAL

    return np.array(points[:n_obs]), np.array(points[n_obs:])


def generate_curved(n_obs, n_pred, speed, direction, curvature, noise_std=0.02):
    """Trajectory with constant turning rate."""
    total = n_obs + n_pred
    points = []
    x, y = random.uniform(-2, 2), random.uniform(-2, 2)
    heading = direction

    for _ in range(total):
        points.append((x + np.random.normal(0, noise_std),
                        y + np.random.normal(0, noise_std)))
        heading += curvature * FRAME_INTERVAL
        x += speed * math.cos(heading) * FRAME_INTERVAL
        y += speed * math.sin(heading) * FRAME_INTERVAL

    return np.array(points[:n_obs]), np.array(points[n_obs:])


def generate_accel(n_obs, n_pred, speed, direction, accel, noise_std=0.02):
    """Trajectory with linear acceleration."""
    total = n_obs + n_pred
    points = []
    x, y = random.uniform(-2, 2), random.uniform(-2, 2)
    current_speed = speed

    for _ in range(total):
        points.append((x + np.random.normal(0, noise_std),
                        y + np.random.normal(0, noise_std)))
        current_speed = max(0.1, current_speed + accel * FRAME_INTERVAL)
        x += current_speed * math.cos(direction) * FRAME_INTERVAL
        y += current_speed * math.sin(direction) * FRAME_INTERVAL

    return np.array(points[:n_obs]), np.array(points[n_obs:])


def generate_stop_and_go(n_obs, n_pred, speed, direction, noise_std=0.02):
    """Trajectory where pedestrian slows down and speeds up."""
    total = n_obs + n_pred
    points = []
    x, y = random.uniform(-2, 2), random.uniform(-2, 2)

    stop_point = random.randint(total // 3, 2 * total // 3)

    for i in range(total):
        points.append((x + np.random.normal(0, noise_std),
                        y + np.random.normal(0, noise_std)))
        dist_to_stop = abs(i - stop_point)
        factor = max(0.1, dist_to_stop / (total / 2))
        current_speed = speed * factor
        x += current_speed * math.cos(direction) * FRAME_INTERVAL
        y += current_speed * math.sin(direction) * FRAME_INTERVAL

    return np.array(points[:n_obs]), np.array(points[n_obs:])


def _direction_label(angle_rad):
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


def format_coords(coords, time_offset=0.0):
    lines = []
    for i, (x, y) in enumerate(coords):
        t = time_offset + i * FRAME_INTERVAL
        lines.append(f"t={t:.1f}s: ({x:.2f}, {y:.2f})")
    return "\n".join(lines)


def generate_analysis(obs, pred):
    """Generate a natural language analysis of the trajectory."""
    obs_vel = np.diff(obs, axis=0)
    pred_vel = np.diff(pred, axis=0)

    obs_speed = np.mean(np.linalg.norm(obs_vel, axis=1)) / FRAME_INTERVAL
    pred_speed = np.mean(np.linalg.norm(pred_vel, axis=1)) / FRAME_INTERVAL

    speed_change = pred_speed - obs_speed
    if abs(speed_change) < 0.1:
        speed_desc = "速度基本保持稳定"
    elif speed_change > 0:
        speed_desc = f"速度略有增加（约+{speed_change:.2f}m/s）"
    else:
        speed_desc = f"速度略有减缓（约{speed_change:.2f}m/s）"

    obs_dir = math.atan2(obs[-1, 1] - obs[0, 1], obs[-1, 0] - obs[0, 0])
    pred_dir = math.atan2(pred[-1, 1] - pred[0, 1], pred[-1, 0] - pred[0, 0])
    dir_change = math.degrees(pred_dir - obs_dir)
    if abs(dir_change) < 10:
        dir_desc = "运动方向基本保持直线"
    elif dir_change > 0:
        dir_desc = f"运动方向向左偏转约{abs(dir_change):.0f}°"
    else:
        dir_desc = f"运动方向向右偏转约{abs(dir_change):.0f}°"

    return speed_desc, dir_desc


# --------------------------- vehicle kinematics ---------------------------
# Vehicles are non-holonomic: they move along their heading and turn via a yaw
# rate (unicycle/bicycle-style integration), unlike the free 2D pedestrian
# models above. Speeds are far higher (urban ~5-15 m/s, highway ~20-40 m/s).

VEHICLE_MANEUVERS = ["highway_cruise", "lane_change", "car_following",
                     "turn", "roundabout"]
LANE_WIDTH = 3.5  # meters


def _veh_integrate(speeds, yaw_rates, heading0, start, noise_std=0.08):
    """Integrate a non-holonomic path from per-step speeds and yaw rates."""
    x, y = start
    heading = heading0
    pts = []
    for k in range(len(speeds)):
        pts.append((x + np.random.normal(0, noise_std),
                    y + np.random.normal(0, noise_std)))
        heading += yaw_rates[k] * FRAME_INTERVAL
        x += speeds[k] * math.cos(heading) * FRAME_INTERVAL
        y += speeds[k] * math.sin(heading) * FRAME_INTERVAL
    return np.array(pts)


def generate_vehicle(maneuver, n_obs, n_pred):
    """Generate a vehicle trajectory (obs, pred) for a given maneuver."""
    total = n_obs + n_pred
    heading0 = random.uniform(0, 2 * math.pi)     # world heading (variety)
    start = (random.uniform(-40, 40), random.uniform(-40, 40))

    if maneuver == "highway_cruise":
        v = random.uniform(22, 40)
        speeds = [max(0.0, v + np.random.normal(0, 0.3)) for _ in range(total)]
        yaws = [np.random.normal(0, 0.003) for _ in range(total)]
    elif maneuver == "lane_change":
        v = random.uniform(16, 33)
        speeds = [max(0.0, v + np.random.normal(0, 0.3)) for _ in range(total)]
        # two opposite yaw pulses -> net ~one lane lateral shift
        s0 = random.randint(2, total - 8)
        w = random.choice([3, 4])
        amp = random.choice([-1, 1]) * (LANE_WIDTH / (v * (w * FRAME_INTERVAL) ** 2)) * 0.5
        yaws = []
        for k in range(total):
            if s0 <= k < s0 + w:
                yaws.append(amp)
            elif s0 + w <= k < s0 + 2 * w:
                yaws.append(-amp)
            else:
                yaws.append(np.random.normal(0, 0.003))
    elif maneuver == "car_following":
        v0 = random.uniform(8, 25)
        # brake then re-accelerate (or vice versa)
        a = random.uniform(0.5, 2.5) * random.choice([-1, 1])
        speeds, v = [], v0
        for k in range(total):
            phase = a if k < total // 2 else -a
            v = min(40.0, max(1.0, v + phase * FRAME_INTERVAL))
            speeds.append(v)
        yaws = [np.random.normal(0, 0.004) for _ in range(total)]
    elif maneuver == "turn":
        v = random.uniform(5, 12)
        speeds = [max(0.0, v + np.random.normal(0, 0.2)) for _ in range(total)]
        # ~60-100 degree turn spread over the trajectory
        total_turn = math.radians(random.uniform(60, 100)) * random.choice([-1, 1])
        rate = total_turn / (total * FRAME_INTERVAL)
        yaws = [rate + np.random.normal(0, 0.01) for _ in range(total)]
    else:  # roundabout
        v = random.uniform(6, 11)
        speeds = [max(0.0, v + np.random.normal(0, 0.2)) for _ in range(total)]
        rate = random.choice([-1, 1]) * random.uniform(0.18, 0.35)
        yaws = [rate + np.random.normal(0, 0.01) for _ in range(total)]

    pts = _veh_integrate(speeds, yaws, heading0, start)
    return pts[:n_obs], pts[n_obs:]


def generate_one_sample(agent_type="vehicle", normalize="none"):
    """Generate a single synthetic trajectory sample (vehicle or pedestrian)."""
    if agent_type == "vehicle":
        maneuver = random.choice(VEHICLE_MANEUVERS)
        obs, pred = generate_vehicle(maneuver, OBS_LENGTH, PRED_LENGTH)
        scene = random.choice(VEHICLE_SCENE_TYPES)
        agent_label = "车辆"
    else:
        motion_type = random.choice(["linear", "curved", "accel", "stop_go"])
        speed = random.uniform(0.5, 2.0)
        direction = random.uniform(0, 2 * math.pi)
        scene = random.choice(SCENE_TYPES)
        if motion_type == "linear":
            obs, pred = generate_linear(OBS_LENGTH, PRED_LENGTH, speed, direction)
        elif motion_type == "curved":
            curvature = random.uniform(-1.5, 1.5)
            obs, pred = generate_curved(OBS_LENGTH, PRED_LENGTH, speed, direction, curvature)
        elif motion_type == "accel":
            accel = random.uniform(-0.3, 0.3)
            obs, pred = generate_accel(OBS_LENGTH, PRED_LENGTH, speed, direction, accel)
        else:
            obs, pred = generate_stop_and_go(OBS_LENGTH, PRED_LENGTH, speed, direction)
        agent_label = "行人"

    # Agent-centric normalization (identity when normalize="none"). Recompute
    # speed/direction from the normalized frame so the prompt matches the coords.
    obs, pred, _ = normalize_obs_pred(obs, pred, normalize)
    avg_speed = round(float(np.mean(np.linalg.norm(np.diff(obs, axis=0), axis=1))
                            / FRAME_INTERVAL), 2)
    heading = math.atan2(obs[-1, 1] - obs[0, 1], obs[-1, 0] - obs[0, 0])
    dir_label = _direction_label(heading)

    pred_text = format_coords(pred.tolist(), time_offset=OBS_LENGTH * FRAME_INTERVAL)
    speed_desc, dir_desc = generate_analysis(obs, pred)

    # Canonical prompt builder (shared with the demo) prevents format drift.
    user_msg = build_user_prompt(obs.tolist(), scene, avg_speed, dir_label,
                                 agent_label=agent_label)

    assistant_msg = (
        f"根据{agent_label}的运动趋势分析：\n"
        f"- {speed_desc}\n"
        f"- {dir_desc}\n\n"
        f"预测轨迹：\n"
        f"{pred_text}"
    )

    st = "vehicle" if agent_type == "vehicle" else "pedestrian"
    return {
        "messages": [
            {"role": "system", "content": system_prompt(st)},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: data/processed/synthetic_sft.jsonl)")
    parser.add_argument("--normalize", choices=MODES, default="none",
                        help="agent-centric coordinate normalization mode")
    parser.add_argument("--agent_type", choices=["vehicle", "pedestrian", "mixed"],
                        default="vehicle",
                        help="agent type to generate (mixed = vehicle-weighted blend)")
    parser.add_argument("--vehicle_ratio", type=float, default=0.75,
                        help="fraction of vehicle samples when agent_type=mixed")
    args = parser.parse_args()
    print(f"Agent type: {args.agent_type} | Normalization: {args.normalize}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output) if args.output else PROC_DIR / "synthetic_sft.jsonl"

    print(f"Generating {args.num_samples} synthetic trajectory samples...")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            if args.agent_type == "mixed":
                at = "vehicle" if random.random() < args.vehicle_ratio else "pedestrian"
            else:
                at = args.agent_type
            sample = generate_one_sample(at, args.normalize)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1}/{args.num_samples}")

    print(f"Saved {args.num_samples} samples to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

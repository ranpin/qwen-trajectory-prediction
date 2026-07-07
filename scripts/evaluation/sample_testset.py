#!/usr/bin/env python3
"""Draw a scene-stratified random subset of a chat-format test set.

Used to pick a manageable, representative evaluation subset from the full
(19k+) ETH/UCY test set. Deterministic given --seed, so the world-frame and
normalized-frame subsets pick the same underlying trajectories.
"""

import json
import random
import argparse
from collections import defaultdict


def scene_of(item):
    user = next(m["content"] for m in item["messages"] if m["role"] == "user")
    return user.split("场景：", 1)[1].split("（", 1)[0].split("\n", 1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--per_scene", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    rows = [json.loads(l) for l in open(args.test_file)]
    by = defaultdict(list)
    for r in rows:
        by[scene_of(r)].append(r)

    sample = []
    for scene, items in sorted(by.items()):
        random.shuffle(items)
        sample.extend(items[:args.per_scene])
    random.shuffle(sample)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"scenes={ {s: len(v) for s, v in sorted(by.items())} }")
    print(f"sampled {len(sample)} ({args.per_scene}/scene) -> {args.output_file}")


if __name__ == "__main__":
    main()

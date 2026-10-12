"""Scripted pick-and-place demo: sort every cube into a bin, record a video, report results.

python scripts/demo_pick_place.py --seed 0 --video outputs/demo.mp4
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from so101_workcell.sim.robot import SO101
from so101_workcell.sim.scene import CUBE_NAMES, randomize_cubes
from so101_workcell.sim.skills import Skills
from so101_workcell.sim.video import VideoRecorder

PLAN = {"cube_red": "bin_left", "cube_green": "bin_right", "cube_blue": "bin_left"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default="outputs/demo.mp4")
    ap.add_argument("--camera", default="front")
    args = ap.parse_args()

    robot = SO101(seed=args.seed)
    rec = VideoRecorder(camera=args.camera)
    skills = Skills(robot, on_tick=rec)
    state = randomize_cubes(robot)
    for n, p in state.cubes.items():
        print(f"{n:11s} at x={p[0]:.3f} y={p[1]:.3f}")
    skills.home()

    t0 = time.time()
    results = []
    for cube in CUBE_NAMES:
        r = skills.pick_and_place(cube, PLAN[cube])
        print(f"  {cube:11s} -> {PLAN[cube]:9s} : {'OK ' if r else 'FAIL'} ({r.reason})")
        results.append(r.ok)
    skills.home()
    wall = time.time() - t0
    print(f"success {sum(results)}/{len(results)}  sim time {robot.time:.1f}s  wall {wall:.1f}s  frames {len(rec.frames)}")
    out = rec.save(args.video)
    print("video ->", out)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

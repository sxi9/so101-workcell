"""Language-driven workcell: instruction -> perception -> planner -> verified skills.

python scripts/run_agent.py --instruction "put the red cube in the left bin" --planner rule
python scripts/run_agent.py --instruction "sort the cubes: red and blue on the left, green on the right" --planner claude
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from so101_workcell.agent.executor import Executor
from so101_workcell.agent.perception import observe, scene_description
from so101_workcell.agent.planner import make_planner
from so101_workcell.sim.robot import SO101
from so101_workcell.sim.scene import randomize_cubes
from so101_workcell.sim.skills import Skills
from so101_workcell.sim.video import VideoRecorder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--planner", default="rule", choices=["rule", "claude"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default="outputs/agent.mp4")
    ap.add_argument("--log", default="outputs/agent_log.json")
    args = ap.parse_args()

    robot = SO101(seed=args.seed)
    rec = VideoRecorder(camera="front")
    skills = Skills(robot, on_tick=rec)
    randomize_cubes(robot)
    skills.home()

    image, dets = observe(robot)
    Image.fromarray(image).save(Path(args.video).with_suffix(".overhead.png"))
    print("perception:", scene_description(dets))

    planner = make_planner(args.planner)
    plan = planner.plan(args.instruction, dets, image)
    print(f"plan ({planner.name}): {plan.reasoning}")
    for s in plan.steps:
        print(f"   {s.action}({s.object} -> {s.target})")
    if not plan.feasible:
        print("plan infeasible:", plan.infeasible_reason)

    log = Executor(robot, skills).run(args.instruction, plan, planner.name)
    print(f"result: {'SUCCESS' if log.ok else 'FAILED'}  sim {log.sim_seconds:.1f}s  wall {log.wall_seconds:.1f}s")
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text(json.dumps(log.to_dict(), indent=2))
    if rec.frames:
        print("video ->", rec.save(args.video))
    return 0 if log.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

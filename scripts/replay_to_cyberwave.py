"""Plan a pick-and-place in local MuJoCo, then stream the same joint targets to a Cyberwave
SO-101 twin (free Playground by default), a cloud MuJoCo sim, or the live arm.

python scripts/replay_to_cyberwave.py --backend dry-run                    # no account needed
python scripts/replay_to_cyberwave.py --backend playground                 # needs CYBERWAVE_API_KEY
python scripts/replay_to_cyberwave.py --backend playground --instruction "put the red cube in the right bin"

The planner/skills run against the local physics (they need contacts to verify the grasp).
What reaches the twin is the 10 Hz joint-target stream, which is exactly what the physical arm
receives, so this is the sim -> twin -> real path with no code change on the control side.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from so101_workcell.agent.executor import Executor
from so101_workcell.agent.perception import observe
from so101_workcell.agent.planner import RulePlanner
from so101_workcell.sim.backends import JointTrajectory, make_backend
from so101_workcell.sim.robot import SO101
from so101_workcell.sim.scene import randomize_cubes
from so101_workcell.sim.skills import CONTROL_HZ, Skills


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dry-run", choices=["dry-run", "playground", "simulation", "live"])
    ap.add_argument("--instruction", default="put every cube in the left bin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=int, default=10, help="replay rate in Hz (targets are sub-sampled from the 50 Hz controller)")
    ap.add_argument("--speed", type=float, default=1.0, help=">1 plays faster than real time")
    ap.add_argument("--save", default="outputs/trajectory.npz")
    args = ap.parse_args()

    every = max(1, CONTROL_HZ // args.rate)
    traj = JointTrajectory(hz=CONTROL_HZ / every)

    # 1. plan + execute locally with verification
    robot = SO101(seed=args.seed)
    skills = Skills(robot, on_tick=traj.record_hook(every))
    randomize_cubes(robot)
    skills.home()
    _img, dets = observe(robot)
    plan = RulePlanner().plan(args.instruction, dets)
    log = Executor(robot, skills).run(args.instruction, plan, "rule")
    print(f"local execution: {'ok' if log.ok else 'FAILED'}; {len(traj.targets)} joint targets at {traj.hz:.0f} Hz")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    traj.save(args.save)
    if not log.ok:
        return 1

    # 2. stream the same targets to the chosen backend
    backend = make_backend(args.backend)
    try:
        n = traj.replay(backend, realtime=args.backend != "dry-run", speed=args.speed)
        print(f"streamed {n} targets to '{args.backend}'")
        if args.backend == "dry-run":
            print("first target:", backend.commands[0].round(3), "last target:", backend.commands[-1].round(3))
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

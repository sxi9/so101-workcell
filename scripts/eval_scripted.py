"""Evaluate the scripted pick-and-place skills over randomized layouts.

python scripts/eval_scripted.py --episodes 20
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from so101_workcell.sim.robot import SO101
from so101_workcell.sim.scene import CUBE_NAMES, randomize_cubes
from so101_workcell.sim.skills import Skills

PLAN = {"cube_red": "bin_left", "cube_green": "bin_right", "cube_blue": "bin_left"}


def run_episode(seed: int) -> list[dict]:
    robot = SO101(seed=seed)
    skills = Skills(robot)
    randomize_cubes(robot)
    skills.home()
    rows = []
    for cube in CUBE_NAMES:
        r = skills.pick_and_place(cube, PLAN[cube])
        rows.append({"seed": seed, "cube": cube, "bin": PLAN[cube], "ok": r.ok, "reason": r.reason})
    robot.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--out", default="outputs/eval_scripted.json")
    args = ap.parse_args()

    t0 = time.time()
    rows: list[dict] = []
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        ep_rows = run_episode(seed)
        rows.extend(ep_rows)
        n_ok = sum(r["ok"] for r in ep_rows)
        print(f"seed {seed}: {n_ok}/{len(ep_rows)}" + ("" if n_ok == len(ep_rows) else "  " + "; ".join(r["reason"] for r in ep_rows if not r["ok"])))
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    failures = Counter(r["reason"] for r in rows if not r["ok"])
    summary = {"episodes": args.episodes, "attempts": n, "success": ok, "rate": ok / n, "failures": dict(failures), "wall_s": time.time() - t0}
    print(json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate a trained SAC policy: success rate with/without domain randomization, optional video.

python scripts/eval_rl.py --model runs/lift_sac/final.zip --episodes 50 --video outputs/rl_lift.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from so101_workcell.rl.env import make_env
from so101_workcell.sim.video import VideoRecorder


def evaluate(model, episodes: int, dr: bool, seed: int, recorder: VideoRecorder | None = None, record_eps: int = 3) -> dict:
    env = make_env(seed=seed, dr=dr)
    succ, grasped, max_z = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            done = term or trunc
            if recorder is not None and ep < record_eps:
                recorder(env.robot)
        succ.append(env.info.success)
        grasped.append(env.info.grasped)
        max_z.append(env.info.max_cube_z)
    env.close()
    return {"episodes": episodes, "dr": dr, "success_rate": float(np.mean(succ)), "grasp_rate": float(np.mean(grasped)), "mean_max_cube_z": float(np.mean(max_z))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/lift_sac/final.zip")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default="outputs/eval_rl.json")
    args = ap.parse_args()
    model = SAC.load(args.model, device="cpu")
    rec = VideoRecorder(camera="front", every=1, fps=10) if args.video else None
    results = {
        "nominal": evaluate(model, args.episodes, dr=False, seed=args.seed, recorder=rec),
        "randomized": evaluate(model, args.episodes, dr=True, seed=args.seed + 1),
    }
    print(json.dumps(results, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    if rec is not None:
        print("video ->", rec.save(args.video))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

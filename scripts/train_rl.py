"""Train SAC on the SO-101 grasp-and-lift environment.

python scripts/train_rl.py --steps 200000 --run lift_sac
Monitor:  tensorboard --logdir runs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from so101_workcell.rl.env import make_env


class SuccessLogger(BaseCallback):
    """Logs the rolling episode success rate to TensorBoard and stdout."""

    def __init__(self, window: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self.hist: list[float] = []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.hist.append(float(info.get("is_success", False)))
        if self.hist and self.n_calls % 2000 == 0:
            rate = float(np.mean(self.hist[-self.window :]))
            self.logger.record("rollout/success_rate", rate)
            if self.verbose:
                print(f"step {self.num_timesteps}: success {rate:.2f} over last {min(self.window, len(self.hist))} eps")
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--run", default="lift_sac")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    def factory(i):
        return lambda: Monitor(make_env(seed=args.seed * 1000 + i, dr=not args.no_dr), info_keywords=("is_success",))

    env = SubprocVecEnv([factory(i) for i in range(args.n_envs)]) if args.n_envs > 1 else DummyVecEnv([factory(0)])

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=400_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log="runs",
        device=args.device,
        seed=args.seed,
        verbose=0,
    )
    t0 = time.time()
    model.learn(
        total_timesteps=args.steps,
        tb_log_name=args.run,
        callback=[SuccessLogger(verbose=1), CheckpointCallback(save_freq=max(25_000 // args.n_envs, 1), save_path=str(run_dir), name_prefix="ckpt")],
        progress_bar=False,
    )
    model.save(run_dir / "final")
    meta = {"steps": args.steps, "n_envs": args.n_envs, "dr": not args.no_dr, "wall_s": time.time() - t0, "device": args.device}
    (run_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("saved", run_dir / "final.zip", json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

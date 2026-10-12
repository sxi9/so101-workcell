"""Upload trained policies, evaluation results and videos to the Hugging Face Hub.

Usage (needs HF_TOKEN with write scope in the environment or `huggingface-cli login`):

    python -m so101_workcell.hub.push model --repo <user>/so101-lift-sac \
        --files runs/lift_sac/final.zip outputs/eval_rl.json outputs/rl_lift.mp4

The model card is generated from the eval JSON so the Hub page shows the numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, upload_file


def model_card(repo: str, eval_json: Path | None, extra: dict | None = None) -> str:
    results = json.loads(eval_json.read_text()) if eval_json and eval_json.exists() else {}
    rows = []
    for name, r in results.items():
        if isinstance(r, dict) and "success_rate" in r:
            rows.append(f"| {name} | {r['episodes']} | {r['success_rate']*100:.0f}% | {r.get('grasp_rate', float('nan'))*100:.0f}% |")
    table = "\n".join(["| condition | episodes | lift success | grasp rate |", "|---|---|---|---|"] + rows) if rows else "_no evaluation attached_"
    extra = extra or {}
    return f"""---
license: apache-2.0
library_name: stable-baselines3
tags:
  - reinforcement-learning
  - robotics
  - mujoco
  - so-101
  - sim-to-real
pipeline_tag: reinforcement-learning
---

# {repo.split('/')[-1]}

SAC policy for grasp-and-lift on the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm,
trained in MuJoCo with domain randomization (mass, size, friction, actuator gain, gravity,
control latency, action and observation noise). Part of the
[so101-workcell](https://github.com/{extra.get('github', '<github-user>/so101-workcell')}) project.

## Results

{table}

## Interface

* **Action** (4): Cartesian fingertip delta (dx, dy, dz) scaled to 2 cm per step, gripper
  command (-1 open, +1 close). Fingers point down; an IK solve maps targets to joint angles.
* **Observation** (23): arm joint positions and velocities, gripper opening, fingertip position,
  cube position, cube minus fingertip, cube z-velocity, previous (dz, gripper) action.
* **Control rate**: 10 Hz. Episodes are 100 steps.

## Usage

```python
from stable_baselines3 import SAC
from so101_workcell.rl.env import make_env

env = make_env(seed=0, dr=True)
model = SAC.load("final.zip")
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(action)
```

Training: {json.dumps(extra.get('train', {}))}
"""


def push_model(repo: str, files: list[str], eval_json: str | None, train_meta: str | None, github: str | None, private: bool) -> str:
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
    extra = {"github": github or "<github-user>/so101-workcell"}
    if train_meta and Path(train_meta).exists():
        extra["train"] = json.loads(Path(train_meta).read_text())
    card = model_card(repo, Path(eval_json) if eval_json else None, extra)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="model")
    for f in files:
        p = Path(f)
        upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=repo, repo_type="model")
    return f"https://huggingface.co/{repo}"


def push_folder(repo: str, folder: str, repo_type: str, private: bool) -> str:
    api = HfApi()
    api.create_repo(repo, repo_type=repo_type, exist_ok=True, private=private)
    api.upload_folder(folder_path=folder, repo_id=repo, repo_type=repo_type)
    return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo}"


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model", help="upload an SB3 policy with an auto-generated model card")
    m.add_argument("--repo", required=True)
    m.add_argument("--files", nargs="+", required=True)
    m.add_argument("--eval-json", default="outputs/eval_rl.json")
    m.add_argument("--train-meta", default="runs/lift_sac/train_meta.json")
    m.add_argument("--github", default=None, help="github user/repo for the card link")
    m.add_argument("--private", action="store_true")
    d = sub.add_parser("dataset", help="upload a folder as a dataset repo (e.g. a LeRobot dataset)")
    d.add_argument("--repo", required=True)
    d.add_argument("--folder", required=True)
    d.add_argument("--private", action="store_true")
    args = ap.parse_args()
    if args.cmd == "model":
        print(push_model(args.repo, args.files, args.eval_json, args.train_meta, args.github, args.private))
    else:
        print(push_folder(args.repo, args.folder, "dataset", args.private))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

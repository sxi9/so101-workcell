"""Record scripted pick-and-place demonstrations as a LeRobot v3 dataset.

The scripted skills act as the "teleoperator": each episode is one language-conditioned
pick-and-place on a randomized layout, sampled at 10 Hz with two cameras. The result is the
same format LeRobot's `lerobot-train` (ACT, SmolVLA, ...) and Cyberwave's dataset import
consume, so the sim demos can seed a policy that is later fine-tuned on real-arm episodes.

    pip install -e ".[lerobot]"
    python -m so101_workcell.hub.lerobot_export --episodes 50 --repo-id <user>/so101_sim_pick_place --push

Features:
    observation.state          (6,) float32  joint positions [rad], gripper last
    action                     (6,) float32  joint position targets [rad] for the next step
    observation.images.front   480x640 video
    observation.images.wrist   480x640 video
    task                       natural-language instruction for the episode
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from so101_workcell.sim.robot import JOINT_NAMES, SO101
from so101_workcell.sim.scene import BIN_NAMES, CUBE_NAMES, randomize_cubes
from so101_workcell.sim.skills import CONTROL_HZ, Skills

FPS = 10
EVERY = CONTROL_HZ // FPS
CAMERAS = {"front": (480, 640), "wrist": (480, 640)}

TASK_TEMPLATES = [
    "put the {color} cube in the {side} bin",
    "move the {color} cube to the {side} bin",
    "place the {color} cube into the {side} bin",
]


def dataset_features() -> dict:
    feats = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
    }
    for cam, (h, w) in CAMERAS.items():
        feats[f"observation.images.{cam}"] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return feats


class FrameSampler:
    """on_tick hook: every EVERY control ticks, capture state, cameras and the joint target."""

    def __init__(self, robot: SO101):
        self.robot = robot
        self.frames: list[dict] = []
        self._n = 0

    def __call__(self, robot: SO101) -> None:
        self._n += 1
        if self._n % EVERY:
            return
        frame = {
            "observation.state": robot.q.astype(np.float32),
            "action": robot.ctrl.astype(np.float32),
        }
        for cam, (h, w) in CAMERAS.items():
            frame[f"observation.images.{cam}"] = robot.render(cam, w, h).copy()
        self.frames.append(frame)


def record_episode(seed: int, rng: np.random.Generator) -> tuple[list[dict], str, bool]:
    robot = SO101(seed=seed)
    sampler = FrameSampler(robot)
    skills = Skills(robot, on_tick=sampler)
    randomize_cubes(robot)
    skills.home()
    cube = CUBE_NAMES[int(rng.integers(len(CUBE_NAMES)))]
    bin_name = BIN_NAMES[int(rng.integers(len(BIN_NAMES)))]
    color, side = cube.split("_")[1], bin_name.split("_")[1]
    task = TASK_TEMPLATES[int(rng.integers(len(TASK_TEMPLATES)))].format(color=color, side=side)
    sampler.frames.clear()  # drop the homing motion; the episode starts parked
    ok = bool(skills.pick_and_place(cube, bin_name))
    skills.home()
    robot.close()
    return sampler.frames, task, ok


def export(root: Path, repo_id: str, episodes: int, seed0: int, *, keep_failed: bool = False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # heavy import, keep lazy

    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=dataset_features(), root=root, robot_type="so101", use_videos=True)
    rng = np.random.default_rng(seed0)
    kept = 0
    for ep in range(episodes):
        frames, task, ok = record_episode(seed0 + ep, rng)
        if not ok and not keep_failed:
            print(f"episode {ep}: skipped (skill failed)")
            continue
        for f in frames:
            ds.add_frame({**f, "task": task})
        ds.save_episode()
        kept += 1
        print(f"episode {ep}: {len(frames)} frames, task='{task}'")
    ds.finalize()
    return ds, kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--repo-id", default="local/so101_sim_pick_place")
    ap.add_argument("--root", default="outputs/lerobot/so101_sim_pick_place")
    ap.add_argument("--push", action="store_true", help="push to the Hugging Face Hub (needs HF_TOKEN)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    ds, kept = export(Path(args.root), args.repo_id, args.episodes, args.seed0)
    print(f"dataset: {kept} episodes, {ds.num_frames} frames -> {args.root}")
    if args.push:
        ds.push_to_hub(private=args.private, tags=["so101", "mujoco", "simulation", "pick-and-place"])
        print(f"pushed -> https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

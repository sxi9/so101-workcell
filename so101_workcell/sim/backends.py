"""Arm backends: the same joint-space commands go to local MuJoCo or to a Cyberwave twin.

The skills layer plans in MuJoCo (it needs contact physics to verify grasps). The resulting
joint-target stream is what a real SO-101 or its Cyberwave digital twin consumes, so a
backend only needs two operations: push joint targets, read joint positions.

    MujocoBackend     - wraps SO101 (local, free, used for everything in this repo)
    CyberwaveBackend  - the-robot-studio/so101 twin on Cyberwave: "playground" (free kinematic
                        sim), "simulation" (cloud MuJoCo, billed) or "live" (real arm)
    DryRunBackend     - logs commands; used in tests and for --dry-run
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .robot import JOINT_NAMES, SO101


class ArmBackend(Protocol):
    joint_names: list[str]

    def set_joint_targets(self, q: np.ndarray) -> None: ...
    def get_joints(self) -> np.ndarray: ...
    def close(self) -> None: ...


class MujocoBackend:
    def __init__(self, robot: SO101):
        self.robot = robot
        self.joint_names = list(JOINT_NAMES)

    def set_joint_targets(self, q: np.ndarray) -> None:
        self.robot.set_ctrl(np.asarray(q)[:5])
        self.robot.data.ctrl[self.robot.act_ids[5]] = float(q[5])

    def get_joints(self) -> np.ndarray:
        return self.robot.q

    def close(self) -> None:
        self.robot.close()


@dataclass
class DryRunBackend:
    joint_names: list[str] = field(default_factory=lambda: list(JOINT_NAMES))
    commands: list[np.ndarray] = field(default_factory=list)

    def set_joint_targets(self, q: np.ndarray) -> None:
        self.commands.append(np.asarray(q, float).copy())

    def get_joints(self) -> np.ndarray:
        return self.commands[-1] if self.commands else np.zeros(6)

    def close(self) -> None:
        pass


class CyberwaveBackend:
    """SO-101 twin on Cyberwave. Needs CYBERWAVE_API_KEY (and optionally
    CYBERWAVE_ENVIRONMENT_ID) in the environment.

    Joint names on the twin are read from the catalog; if they differ from our MJCF names
    they are matched by order (both are the standard 6-joint SO-101 chain).
    """

    def __init__(self, mode: str = "playground", asset: str = "the-robot-studio/so101", environment_id: str | None = None):
        from cyberwave import Cyberwave  # optional at import time

        self.cw = Cyberwave(environment_id=environment_id) if environment_id else Cyberwave()
        self.twin = self.cw.twin(asset)
        self.cw.affect(mode)
        self.twin.policy.ensure_attached()
        names = list(self.twin.joints.list() or [])
        if len(names) != len(JOINT_NAMES):
            raise RuntimeError(f"twin exposes {len(names)} joints {names}, expected {len(JOINT_NAMES)}")
        self.joint_names = names if set(names) == set(JOINT_NAMES) else names  # positional mapping either way
        self._order = [names.index(n) for n in JOINT_NAMES] if set(names) == set(JOINT_NAMES) else list(range(len(names)))

    def set_joint_targets(self, q: np.ndarray) -> None:
        q = np.asarray(q, float)
        values = {self.joint_names[self._order[i]]: float(q[i]) for i in range(len(JOINT_NAMES))}
        self.twin.joints.set(values, what_data="position", degrees=False)

    def get_joints(self) -> np.ndarray:
        state = self.twin.joints.get_all()
        return np.array([float(state.get(self.joint_names[self._order[i]], 0.0)) for i in range(len(JOINT_NAMES))])

    def close(self) -> None:
        pass


@dataclass
class JointTrajectory:
    """Joint targets sampled at a fixed rate, as produced by the skills layer."""
    hz: float
    targets: list[np.ndarray] = field(default_factory=list)

    def record_hook(self, every: int = 1):
        """Return an on_tick hook that samples the current joint targets."""
        counter = {"n": 0}

        def hook(robot: SO101) -> None:
            counter["n"] += 1
            if counter["n"] % every == 0:
                self.targets.append(robot.ctrl.copy())

        return hook

    def replay(self, backend: ArmBackend, *, realtime: bool = True, speed: float = 1.0) -> int:
        dt = 1.0 / (self.hz * speed)
        t0 = time.time()
        for i, q in enumerate(self.targets):
            backend.set_joint_targets(q)
            if realtime:
                time.sleep(max(0.0, t0 + (i + 1) * dt - time.time()))
        return len(self.targets)

    def save(self, path) -> None:
        np.savez_compressed(path, hz=self.hz, targets=np.array(self.targets), joint_names=np.array(JOINT_NAMES))

    @staticmethod
    def load(path) -> "JointTrajectory":
        d = np.load(path)
        return JointTrajectory(float(d["hz"]), list(d["targets"]))


def make_backend(name: str, robot: SO101 | None = None, **kw) -> ArmBackend:
    if name == "mujoco":
        return MujocoBackend(robot or SO101())
    if name == "dry-run":
        return DryRunBackend()
    if name in ("playground", "simulation", "live"):
        return CyberwaveBackend(mode=name, **kw)
    raise ValueError(f"unknown backend {name!r}")

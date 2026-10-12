"""Workcell scene description and randomization: cubes, bins, workspace bounds."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .robot import SO101

CUBE_SIZE = 0.025  # edge length, metres
CUBE_NAMES = ["cube_red", "cube_green", "cube_blue"]
BIN_NAMES = ["bin_left", "bin_right"]
BIN_HALF = 0.045

# Region in front of the arm where cubes spawn (world frame, metres). Fingers-down grasps are
# reachable out to a radius of ~0.27 m at table height (see scripts/measure_geometry.py).
SPAWN_X = (0.13, 0.25)
SPAWN_Y = (-0.09, 0.09)
MIN_CUBE_GAP = 0.05


@dataclass
class SceneState:
    cubes: dict[str, np.ndarray] = field(default_factory=dict)
    bins: dict[str, np.ndarray] = field(default_factory=dict)

    @staticmethod
    def read(robot: SO101) -> "SceneState":
        return SceneState(
            cubes={n: robot.body_pos(n) for n in CUBE_NAMES},
            bins={n: robot.body_pos(n) for n in BIN_NAMES},
        )


def randomize_cubes(robot: SO101, rng: np.random.Generator | None = None, *, yaw: bool = True) -> SceneState:
    """Scatter cubes on the table without overlap. Returns the resulting scene state."""
    rng = rng or robot.rng
    placed: list[np.ndarray] = []
    for name in CUBE_NAMES:
        for _ in range(200):
            xy = np.array([rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y)])
            if all(np.linalg.norm(xy - p) >= MIN_CUBE_GAP for p in placed):
                break
        placed.append(xy)
        angle = rng.uniform(-np.pi / 4, np.pi / 4) if yaw else 0.0
        quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        robot.set_free_body_pose(name, [xy[0], xy[1], CUBE_SIZE / 2 + 0.001], quat)
    # settle contacts
    robot.step(50)
    return SceneState.read(robot)


def cube_yaw(robot: SO101, name: str) -> float:
    """Yaw of a cube about world z, folded into [-pi/4, pi/4] (cube symmetry)."""
    w, x, y, z = robot.body_quat(name)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(yaw)


def in_bin(robot: SO101, cube: str, bin_name: str, *, margin: float = 0.0) -> bool:
    p = robot.body_pos(cube)
    b = robot.body_pos(bin_name)
    inside_xy = np.all(np.abs(p[:2] - b[:2]) <= BIN_HALF - margin)
    on_floor = p[2] < 0.05
    return bool(inside_xy and on_floor)


def is_lifted(robot: SO101, cube: str, height: float = 0.05) -> bool:
    return bool(robot.body_pos(cube)[2] > height)

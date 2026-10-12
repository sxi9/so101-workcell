"""Scripted manipulation skills for the SO-101 workcell.

Each skill is a small closed-loop routine: plan with IK, execute with the position
controllers, then *verify* the outcome from simulator state. Skills return a SkillResult so
the agent layer can retry or re-plan instead of trusting open-loop execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .ik import solve_ik
from .robot import SO101
from .scene import CUBE_SIZE, clearance, cube_yaw, in_bin, is_lifted

FrameHook = Callable[[SO101], None]  # called every control tick (for video recording)

# IK rest posture: fingertips at (0.22, 0, 0.09) pointing straight down (solved offline).
REST_Q = np.array([0.043, -0.097, 0.003, 1.658, -1.479])
HOME_Q = np.array([0.0, -1.7, 0.0, 0.0, 0.0])  # arm straight up: out of the overhead camera view

# When a straight-down approach is out of reach, retry with the fingers tilted away from the
# base by these angles (degrees). Tilting extends the reach at height.
TILT_FALLBACKS = (0.0, 15.0, 30.0, 45.0)

CONTROL_HZ = 50
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * 0.002)))

# Fingertip pad geometry relative to the gripperframe site (measured, see scripts/measure_geometry.py):
#   fixed pad inner face sits 0.9 mm along +close axis from the site
#   both pads span roughly -8..+10 mm along the finger axis (positive = toward the fingertips)
PAD_FACE_OFFSET = 0.0009
GRASP_CLEARANCE = 0.004  # gap between the fixed pad and the cube face before closing
GRASP_HEIGHT_OFFSET = 0.003  # site height above the cube centre so the pads straddle the centre


@dataclass
class SkillResult:
    name: str
    ok: bool
    reason: str = ""
    info: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


class Skills:
    def __init__(self, robot: SO101, *, on_tick: FrameHook | None = None):
        self.robot = robot
        self.on_tick = on_tick
        self._q_target = robot.q[:5].copy()

    # ---- primitives --------------------------------------------------------------------
    def _tick(self) -> None:
        self.robot.step(SUBSTEPS)
        if self.on_tick:
            self.on_tick(self.robot)

    def settle(self, seconds: float = 0.3) -> None:
        for _ in range(int(seconds * CONTROL_HZ)):
            self._tick()

    def move_joints(self, q_goal: np.ndarray, *, duration: float = 1.0, tol: float = 0.03) -> SkillResult:
        """Interpolate joint targets linearly over `duration`, then wait for convergence."""
        q_goal = np.asarray(q_goal, float)
        q0 = self._q_target.copy()
        n = max(1, int(duration * CONTROL_HZ))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)  # smoothstep
            self._q_target = q0 + s * (q_goal - q0)
            self.robot.set_ctrl(self._q_target)
            self._tick()
        for _ in range(int(0.6 * CONTROL_HZ)):
            if np.max(np.abs(self.robot.q[:5] - q_goal)) < tol:
                break
            self._tick()
        err = float(np.max(np.abs(self.robot.q[:5] - q_goal)))
        return SkillResult("move_joints", err < tol * 2, f"joint err {err:.3f}", {"err": err})

    def plan_to(self, pos, *, close_dir=None, max_tilt: float = 45.0, signed: bool = True):
        """IK with tilt fallbacks. Returns (IKResult, tilt_deg) for the first reachable tilt."""
        pos = np.asarray(pos, float)
        radial = np.array([pos[0], pos[1], 0.0])
        radial /= max(np.linalg.norm(radial), 1e-6)
        best = None
        for tilt in TILT_FALLBACKS:
            if tilt > max_tilt:
                break
            t = np.radians(tilt)
            fdir = radial * np.sin(t) + np.array([0, 0, -np.cos(t)])
            ik = solve_ik(self.robot, pos, fdir, close_dir, q_init=self._q_target, q_rest=REST_Q, close_dir_signed=signed)
            if ik.ok:
                return ik, tilt
            if best is None or ik.pos_err < best[0].pos_err:
                best = (ik, tilt)
        return best

    def move_to(self, pos, *, close_dir=None, duration: float = 1.0, max_tilt: float = 45.0, signed: bool = True) -> SkillResult:
        """Move the fingertip site to a world position, fingers down (tilting if needed)."""
        ik, tilt = self.plan_to(pos, close_dir=close_dir, max_tilt=max_tilt, signed=signed)
        if not ik.ok:
            return SkillResult("move_to", False, f"IK failed: pos {ik.pos_err*1000:.1f} mm, ori {ik.ori_err:.2f}", {"ik": ik})
        res = self.move_joints(ik.q, duration=duration)
        p = self.robot.ee_pos()
        err = float(np.linalg.norm(p - np.asarray(pos)))
        ok = res.ok and err < 0.012
        return SkillResult("move_to", ok, f"ee err {err*1000:.1f} mm, tilt {tilt:.0f}", {"ee_err": err, "ik": ik, "tilt": tilt})

    def move_linear(self, pos, *, close_dir=None, duration: float = 0.8, max_tilt: float = 15.0, step_m: float = 0.01, signed: bool = True) -> SkillResult:
        """Straight-line Cartesian move: IK on evenly spaced waypoints so the fingertips do not
        swing sideways on the way down (a joint-space interpolation does not stay vertical)."""
        pos = np.asarray(pos, float)
        start = self.robot.ee_pos()
        dist = float(np.linalg.norm(pos - start))
        n = max(2, int(np.ceil(dist / step_m)))
        seg_t = max(duration / n, 1.0 / CONTROL_HZ)
        for i in range(1, n + 1):
            wp = start + (pos - start) * (i / n)
            ik, _tilt = self.plan_to(wp, close_dir=close_dir, max_tilt=max_tilt, signed=signed)
            if not ik.ok:
                return SkillResult("move_linear", False, f"IK failed at waypoint {i}/{n}: pos {ik.pos_err*1000:.1f} mm", {"ik": ik})
            self.move_joints(ik.q, duration=seg_t, tol=0.05)
        for _ in range(int(0.3 * CONTROL_HZ)):
            if np.linalg.norm(self.robot.ee_pos() - pos) < 0.004:
                break
            self._tick()
        err = float(np.linalg.norm(self.robot.ee_pos() - pos))
        return SkillResult("move_linear", err < 0.012, f"ee err {err*1000:.1f} mm", {"ee_err": err})

    def gripper(self, opening: float, *, seconds: float = 0.5) -> None:
        self.robot.set_gripper(opening)
        self.settle(seconds)

    def home(self) -> SkillResult:
        self.robot.set_gripper(1.0)
        return self.move_joints(HOME_Q, duration=1.2)

    # ---- grasp geometry ----------------------------------------------------------------
    # The open moving jaw sweeps through this region (site frame: along +close, metres) at
    # roughly table height while closing, so it must be free of cubes and bin walls.
    JAW_SWEEP = (0.03, 0.05, 0.07, 0.09)

    def _grasp_candidates(self, cube: str) -> list[tuple[np.ndarray, float]]:
        """Candidate closing directions for a top-down grasp, best first.

        close_dir points from the fixed jaw to the moving jaw. Candidates are the four
        horizontal face normals of the cube; they are ranked by the clearance of the moving
        jaw's sweep from other objects, with a small preference for a tangential direction
        (which keeps wrist_roll small on this arm).
        """
        c = self.robot.body_pos(cube)
        yaw = cube_yaw(self.robot, cube)
        radial = np.array([c[0], c[1], 0.0])
        radial /= max(np.linalg.norm(radial), 1e-6)
        tangential = np.array([-radial[1], radial[0], 0.0])
        scored = []
        for a in (yaw, yaw + np.pi / 2, yaw + np.pi, yaw + 3 * np.pi / 2):
            d = np.array([np.cos(a), np.sin(a), 0.0])
            clear = min(clearance(self.robot, c[:2] + d[:2] * s, exclude=cube) for s in self.JAW_SWEEP)
            # the fixed jaw sits just outside the opposite face; it must not land on a wall
            clear = min(clear, clearance(self.robot, c[:2] - d[:2] * 0.024, exclude=cube) + 0.02)
            score = min(clear, 0.04) + 0.01 * abs(d @ tangential)
            scored.append((d, score))
        scored.sort(key=lambda t: -t[1])
        return scored

    def _grasp_frame(self, cube: str) -> tuple[np.ndarray, np.ndarray]:
        return self._grasp_candidates(cube)[0][0], self.robot.body_pos(cube)

    def _actual_close_dir(self) -> np.ndarray:
        """Horizontal projection of the site z-axis: fixed pad -> moving pad."""
        d = self.robot.ee_rot()[:, 2].copy()
        d[2] = 0.0
        return d / max(np.linalg.norm(d), 1e-6)

    def _site_for_cube_center(self, center: np.ndarray, close_dir: np.ndarray, z: float) -> np.ndarray:
        """Site position that puts the fixed pad just outside the cube face nearest to it."""
        offset = CUBE_SIZE / 2 + PAD_FACE_OFFSET + GRASP_CLEARANCE
        return np.array([center[0], center[1], z]) - close_dir * offset

    # ---- composite skills --------------------------------------------------------------
    def pick(self, cube: str, *, hover: float = 0.045) -> SkillResult:
        """Top-down grasp of a cube. Verifies the cube is lifted at the end."""
        c = self.robot.body_pos(cube)
        z_grasp = c[2] + GRASP_HEIGHT_OFFSET
        self.robot.set_gripper(1.0)
        # 1. hover above (grasps stay near vertical: at most a small tilt). Try the closing
        #    directions best-first; a direction can be unreachable for wrist_roll.
        r = None
        for close_dir, _score in self._grasp_candidates(cube):
            hover_pos = self._site_for_cube_center(c, close_dir, z_grasp + hover)
            ik, _tilt = self.plan_to(hover_pos, close_dir=close_dir, max_tilt=15)
            if not ik.ok:
                continue
            r = self.move_to(hover_pos, close_dir=close_dir, duration=1.0, max_tilt=15)
            if r:
                break
        if r is None:
            return SkillResult("pick", False, "hover: no reachable grasp direction")
        if not r:
            return SkillResult("pick", False, "hover: " + r.reason, r.info)
        # The IK treats the closing axis as sign-free, so read back the achieved site z-axis
        # (which points from the fixed pad toward the moving pad) and re-derive the offset
        # from it. Never flip it: the pad geometry is attached to the actual axis.
        close_dir_actual = self._actual_close_dir()
        c = self.robot.body_pos(cube)
        hover2 = self._site_for_cube_center(c, close_dir_actual, z_grasp + hover)
        if np.linalg.norm(hover2 - self.robot.ee_pos()) > 0.005:
            r = self.move_to(hover2, close_dir=close_dir_actual, duration=0.6, max_tilt=15)
            if not r:
                return SkillResult("pick", False, "re-hover: " + r.reason, r.info)
            close_dir_actual = self._actual_close_dir()
        # 2. descend
        r = self.move_linear(self._site_for_cube_center(c, close_dir_actual, z_grasp), close_dir=close_dir_actual, duration=0.8)
        if not r:
            return SkillResult("pick", False, "descend: " + r.reason, r.info)
        # 3. close
        self.gripper(0.0, seconds=0.6)
        # 4. lift straight up
        r = self.move_linear(self._site_for_cube_center(c, close_dir_actual, z_grasp + hover), close_dir=close_dir_actual, duration=0.8)
        self.settle(0.2)
        lifted = is_lifted(self.robot, cube, height=0.04)
        return SkillResult("pick", lifted, "lifted" if lifted else "cube not lifted (grasp slipped?)", {"cube": cube})

    def place(self, bin_name: str, cube: str | None = None, *, hover: float = 0.045) -> SkillResult:
        """Carry the held object over a bin, lower, release. Verifies the cube landed inside."""
        b = self.robot.body_pos(bin_name)
        close_dir = self._actual_close_dir()
        target_hi = np.array([b[0], b[1], 0.03 + hover])
        target_lo = np.array([b[0], b[1], 0.055])
        # The cube is held between the pads, so the closing axis is unconstrained during
        # transport; that keeps wrist_roll inside its range over the bins.
        r = self.move_to(target_hi, close_dir=None, duration=1.4)
        if not r:
            return SkillResult("place", False, "transport: " + r.reason, r.info)
        close_dir = self._actual_close_dir()
        r = self.move_linear(target_lo, close_dir=close_dir, duration=0.8, max_tilt=45)
        if not r:
            return SkillResult("place", False, "lower: " + r.reason, r.info)
        self.gripper(1.0, seconds=0.5)
        r = self.move_linear(target_hi, close_dir=close_dir, duration=0.8, max_tilt=45)
        self.settle(0.5)
        if cube is None:
            return SkillResult("place", True, "released")
        ok = in_bin(self.robot, cube, bin_name)
        return SkillResult("place", ok, "in bin" if ok else "cube missed the bin", {"cube": cube, "bin": bin_name})

    def pick_and_place(self, cube: str, bin_name: str) -> SkillResult:
        r = self.pick(cube)
        if not r:
            return r
        r = self.place(bin_name, cube)
        return SkillResult("pick_and_place", r.ok, r.reason, r.info)

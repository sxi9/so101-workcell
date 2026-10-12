"""Thin wrapper around the SO-101 MuJoCo model: named joints, control, FK, rendering helpers."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"
DEFAULT_XML = ASSETS / "workcell" / "workcell.xml"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINT_NAMES[:5]

# Site between the fingertips. Measured at the home pose (all joints zero):
#   x-axis: along the fingers (pointing direction)
#   z-axis: closing axis (the moving jaw swings along it)
#   y-axis: the remaining horizontal axis (jaw width direction)
GRIPPER_SITE = "gripperframe"

# Gripper joint limits in the new calibration are about [-0.17, 1.74] rad. Filled in from the
# measurement script; see docs/geometry.md.
GRIPPER_CLOSED = -0.10
GRIPPER_OPEN = 1.00


class SO101:
    """Owns an MjModel/MjData pair and exposes joint-space control of the arm."""

    def __init__(self, xml_path: str | Path | None = None, *, seed: int | None = None):
        xml_path = Path(xml_path) if xml_path else DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.jnt_ids = np.array([self.model.joint(n).id for n in JOINT_NAMES])
        self.qpos_idx = np.array([self.model.jnt_qposadr[j] for j in self.jnt_ids])
        self.qvel_idx = np.array([self.model.jnt_dofadr[j] for j in self.jnt_ids])
        self.act_ids = np.array([self.model.actuator(n).id for n in JOINT_NAMES])
        self.site_id = self.model.site(GRIPPER_SITE).id
        self.joint_low = self.model.jnt_range[self.jnt_ids, 0]
        self.joint_high = self.model.jnt_range[self.jnt_ids, 1]
        self.dt = self.model.opt.timestep
        self._renderer: mujoco.Renderer | None = None
        self._scratch: mujoco.MjData | None = None

    # ---- state -------------------------------------------------------------------------
    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    @property
    def qd(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    @property
    def ctrl(self) -> np.ndarray:
        return self.data.ctrl[self.act_ids].copy()

    def set_q(self, q: np.ndarray, *, also_ctrl: bool = True) -> None:
        """Teleport joints (no dynamics). Use for resets, not control."""
        self.data.qpos[self.qpos_idx] = q
        self.data.qvel[self.qvel_idx] = 0
        if also_ctrl:
            self.data.ctrl[self.act_ids] = q
        mujoco.mj_forward(self.model, self.data)

    def set_ctrl(self, q_target: np.ndarray) -> None:
        q_target = np.asarray(q_target, dtype=float)
        n = len(q_target)
        self.data.ctrl[self.act_ids[:n]] = np.clip(q_target, self.joint_low[:n], self.joint_high[:n])

    def set_gripper(self, opening: float) -> None:
        """opening in [0, 1]: 0 closed, 1 open."""
        o = float(np.clip(opening, 0.0, 1.0))
        self.data.ctrl[self.act_ids[5]] = GRIPPER_CLOSED + o * (GRIPPER_OPEN - GRIPPER_CLOSED)

    def gripper_opening(self) -> float:
        return float((self.q[5] - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED))

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    # ---- kinematics ------------------------------------------------------------------
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def ee_rot(self) -> np.ndarray:
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def fk(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics for the 5 arm joints without touching the live simulation state."""
        d = self.scratch_data()
        d.qpos[:] = self.data.qpos
        d.qpos[self.qpos_idx[:5]] = q_arm
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.site_id].copy(), d.site_xmat[self.site_id].reshape(3, 3).copy()

    def scratch_data(self) -> mujoco.MjData:
        if self._scratch is None:
            self._scratch = mujoco.MjData(self.model)
        return self._scratch

    # ---- objects -----------------------------------------------------------------------
    def body_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def body_quat(self, name: str) -> np.ndarray:
        return self.data.body(name).xquat.copy()

    def set_free_body_pose(self, name: str, pos, quat=None) -> None:
        """Place a free body (one with a freejoint) at pos with optional wxyz quaternion."""
        jid = self.model.body(name).jntadr[0]
        adr = self.model.jnt_qposadr[jid]
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat if quat is not None else [1, 0, 0, 0]
        vadr = self.model.jnt_dofadr[jid]
        self.data.qvel[vadr : vadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    # ---- rendering ---------------------------------------------------------------------
    def render(self, camera: str = "overhead", width: int = 640, height: int = 480) -> np.ndarray:
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

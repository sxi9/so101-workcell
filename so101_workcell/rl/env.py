"""Gymnasium environment: SO-101 grasp-and-lift a cube in MuJoCo with domain randomization.

Action (4,): Cartesian fingertip delta (dx, dy, dz) in [-1, 1] scaled to ACTION_SCALE metres,
            plus a gripper command in [-1, 1] (-1 open, +1 close). Fingers stay pointing down;
            an IK solve converts the Cartesian target to joint targets each control step.
Observation (26,): arm joint positions (5), arm joint velocities (5), gripper opening (1),
            fingertip position (3), cube position (3), cube - fingertip (3),
            cube z-velocity (1), previous action (2: dz, gripper), commanded target - fingertip (3).
Reward: dense shaping toward the cube, a grasp bonus when the cube sits between the pads with
            the gripper closed, lift reward proportional to cube height, success bonus.
Episode: 100 control steps at 10 Hz. Success = cube lifted above LIFT_SUCCESS metres.

Domain randomization (per reset): cube mass and size, pad/cube/table friction, actuator gain
            scale, gravity, control latency (0-1 step action delay), action noise, cube spawn
            and yaw, initial arm posture jitter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101_workcell.sim.ik import solve_ik
from so101_workcell.sim.robot import SO101
from so101_workcell.sim.scene import CUBE_SIZE, SPAWN_X, SPAWN_Y
from so101_workcell.sim.skills import GRASP_CLEARANCE, GRASP_HEIGHT_OFFSET, PAD_FACE_OFFSET, REST_Q

CUBE = "cube_red"
PARKED = {"cube_green": (-0.25, 0.25), "cube_blue": (-0.25, -0.25)}  # out of the way
CONTROL_DT = 0.1
SUBSTEPS = int(round(CONTROL_DT / 0.002))
ACTION_SCALE = np.array([0.02, 0.02, 0.02])  # metres per step at |a| = 1
LIFT_SUCCESS = 0.06
MAX_STEPS = 100
WORKSPACE_LO = np.array([0.10, -0.14, 0.012])
WORKSPACE_HI = np.array([0.28, 0.14, 0.10])  # fingers-down poses above ~10 cm are out of reach


@dataclass
class DRConfig:
    enabled: bool = True
    mass_range: tuple[float, float] = (0.6, 1.6)  # scale
    size_range: tuple[float, float] = (0.9, 1.1)  # scale of the cube half-size
    friction_range: tuple[float, float] = (0.6, 1.4)  # scale, pads and cube
    gain_range: tuple[float, float] = (0.8, 1.2)  # actuator kp scale
    gravity_range: tuple[float, float] = (-10.3, -9.3)
    latency_steps: tuple[int, int] = (0, 1)
    action_noise: float = 0.05
    obs_noise: float = 0.002
    init_jitter: float = 0.05  # rad


@dataclass
class EpisodeInfo:
    success: bool = False
    grasped: bool = False
    max_cube_z: float = 0.0
    dr: dict = field(default_factory=dict)


class SO101LiftEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, *, dr: DRConfig | None = None, render_mode: str | None = None, seed: int | None = None, camera: str = "front"):
        super().__init__()
        self.robot = SO101(seed=seed)
        self.dr = dr or DRConfig()
        self.render_mode = render_mode
        self.camera = camera
        self.rng = np.random.default_rng(seed)
        m = self.robot.model
        self._nominal = {
            "body_mass": m.body_mass.copy(),
            "body_inertia": m.body_inertia.copy(),
            "geom_size": m.geom_size.copy(),
            "geom_friction": m.geom_friction.copy(),
            "gainprm": m.actuator_gainprm.copy(),
            "biasprm": m.actuator_biasprm.copy(),
            "gravity": m.opt.gravity.copy(),
        }
        self._cube_geom = m.geom(CUBE + "_geom").id
        self._cube_body = m.body(CUBE).id
        self._pad_geoms = [m.geom("pad_fixed").id, m.geom("pad_moving").id]
        self._table_geom = m.geom("table").id
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32)
        self._target = np.zeros(3)
        self._close_dir = np.array([0.0, 1.0, 0.0])
        self._q_target = REST_Q.copy()
        self._prev_action = np.zeros(4)
        self._pending: list[np.ndarray] = []
        self._latency = 0
        self._steps = 0
        self.info = EpisodeInfo()

    # ---- domain randomization ----------------------------------------------------------
    def _apply_dr(self) -> dict:
        m = self.robot.model
        n = self._nominal
        m.body_mass[:] = n["body_mass"]
        m.body_inertia[:] = n["body_inertia"]
        m.geom_size[:] = n["geom_size"]
        m.geom_friction[:] = n["geom_friction"]
        m.actuator_gainprm[:] = n["gainprm"]
        m.actuator_biasprm[:] = n["biasprm"]
        m.opt.gravity[:] = n["gravity"]
        self._latency = 0
        if not self.dr.enabled:
            return {}
        r = self.rng
        ms = r.uniform(*self.dr.mass_range)
        ss = r.uniform(*self.dr.size_range)
        fs = r.uniform(*self.dr.friction_range)
        gs = r.uniform(*self.dr.gain_range)
        m.body_mass[self._cube_body] = n["body_mass"][self._cube_body] * ms
        m.body_inertia[self._cube_body] = n["body_inertia"][self._cube_body] * ms * ss * ss
        m.geom_size[self._cube_geom] = n["geom_size"][self._cube_geom] * ss
        for g in [self._cube_geom, self._table_geom] + self._pad_geoms:
            m.geom_friction[g, 0] = n["geom_friction"][g, 0] * fs
        # position actuators: gainprm[0] = kp, biasprm[1] = -kp
        m.actuator_gainprm[:5, 0] = n["gainprm"][:5, 0] * gs
        m.actuator_biasprm[:5, 1] = n["biasprm"][:5, 1] * gs
        m.opt.gravity[2] = r.uniform(*self.dr.gravity_range)
        self._latency = int(r.integers(self.dr.latency_steps[0], self.dr.latency_steps[1] + 1))
        return {"mass": ms, "size": ss, "friction": fs, "gain": gs, "gravity": float(m.opt.gravity[2]), "latency": self._latency}

    # ---- gym API -------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rb = self.robot
        mujoco.mj_resetData(rb.model, rb.data)
        dr = self._apply_dr()
        # park the unused cubes
        for name, (x, y) in PARKED.items():
            rb.set_free_body_pose(name, [x, y, CUBE_SIZE / 2])
        # spawn the target cube
        xy = np.array([self.rng.uniform(*SPAWN_X), self.rng.uniform(*SPAWN_Y)])
        yaw = self.rng.uniform(-np.pi / 4, np.pi / 4)
        half = float(rb.model.geom_size[self._cube_geom, 2])
        rb.set_free_body_pose(CUBE, [xy[0], xy[1], half + 0.001], [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        # arm starts above the workspace, fingers down, gripper open, with jitter
        q0 = REST_Q + self.rng.uniform(-self.dr.init_jitter, self.dr.init_jitter, 5) * self.dr.enabled
        rb.set_q(np.concatenate([q0, [1.0]]))
        rb.set_gripper(1.0)
        rb.step(25)
        self._q_target = rb.q[:5].copy()
        self._target = rb.ee_pos()
        # Closing axis (fixed pad -> moving pad): tangential to the arm's radial direction at
        # the cube, with the sign the arm can actually reach from its start posture. The IK
        # then holds this signed axis for the whole episode so the grasp-site offset used by
        # the reward matches where the fixed pad really is.
        c = rb.body_pos(CUBE)
        radial = np.array([c[0], c[1], 0.0]) / max(np.linalg.norm(c[:2]), 1e-6)
        tang = np.array([-radial[1], radial[0], 0.0])
        actual = rb.ee_rot()[:, 2].copy()
        actual[2] = 0.0
        self._close_dir = tang if actual @ tang >= 0 else -tang
        ik = solve_ik(rb, self._target, (0, 0, -1), self._close_dir, q_init=self._q_target, q_rest=REST_Q, close_dir_signed=True)
        self._q_target = ik.q
        rb.set_ctrl(self._q_target)
        rb.step(25)
        self._prev_action = np.zeros(4)
        self._pending = []
        self._steps = 0
        self._cube0 = rb.body_pos(CUBE)[:2].copy()
        self.info = EpisodeInfo(dr=dr)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        if self.dr.enabled and self.dr.action_noise > 0:
            action = np.clip(action + self.rng.normal(0, self.dr.action_noise, 4), -1, 1)
        # control latency: delay actions by _latency steps
        self._pending.append(action)
        act = self._pending.pop(0) if len(self._pending) > self._latency else np.zeros(4)
        rb = self.robot
        self._target = np.clip(self._target + act[:3] * ACTION_SCALE, WORKSPACE_LO, WORKSPACE_HI)
        ik = solve_ik(rb, self._target, (0, 0, -1), self._close_dir, q_init=self._q_target, q_rest=REST_Q, max_iters=30, close_dir_signed=True)
        self._q_target = ik.q
        rb.set_ctrl(self._q_target)
        rb.set_gripper(0.5 - 0.5 * act[3])  # +1 -> closed
        rb.step(SUBSTEPS)
        self._steps += 1
        self._prev_action = act
        obs = self._obs()
        reward, terms = self._reward()
        cube_z = rb.body_pos(CUBE)[2]
        self.info.max_cube_z = max(self.info.max_cube_z, float(cube_z))
        success = cube_z > LIFT_SUCCESS and terms["grasped"]
        self.info.success = self.info.success or success
        terminated = False
        truncated = self._steps >= MAX_STEPS
        info = {"success": success, "is_success": self.info.success, **terms}
        return obs, float(reward), terminated, truncated, info

    # ---- helpers -----------------------------------------------------------------------
    def _cube_between_pads(self) -> bool:
        """True when both pads are in contact with the cube."""
        m, d = self.robot.model, self.robot.data
        touched = set()
        for i in range(d.ncon):
            c = d.contact[i]
            pair = (c.geom1, c.geom2)
            if self._cube_geom in pair:
                other = pair[1] if pair[0] == self._cube_geom else pair[0]
                if other in self._pad_geoms:
                    touched.add(other)
        return len(touched) == 2

    def _reward(self) -> tuple[float, dict]:
        rb = self.robot
        ee = rb.ee_pos()
        c = rb.body_pos(CUBE)
        # grasp point: the fixed pad should sit just outside the cube face on the -close side
        grasp_site = c - self._close_dir * (CUBE_SIZE / 2 + PAD_FACE_OFFSET + GRASP_CLEARANCE / 2)
        grasp_site[2] = c[2] + GRASP_HEIGHT_OFFSET
        dist = float(np.linalg.norm(ee - grasp_site))
        grasped = self._cube_between_pads() and rb.gripper_opening() < 0.6
        lift = float(np.clip((c[2] - CUBE_SIZE / 2) / LIFT_SUCCESS, 0, 1.5))
        # reach: a linear term keeps a constant gradient far from the cube (tanh alone is flat
        # there, which let an early policy settle at the workspace boundary), plus a sharp
        # bonus near the grasp site
        r_reach = -4.0 * dist + (1.0 - np.tanh(10.0 * dist))
        r_grasp = 1.0 if grasped else 0.0
        r_lift = 2.0 * lift if grasped else 0.0
        r_success = 5.0 if (grasped and c[2] > LIFT_SUCCESS) else 0.0
        # penalties: closing far from the cube; shoving the cube around instead of grasping it
        # (an early policy learned to drag the cube into a workspace corner)
        r_pen = -0.1 if (rb.gripper_opening() < 0.6 and dist > 0.03 and not grasped) else 0.0
        if not grasped:
            r_pen -= 3.0 * float(min(np.linalg.norm(c[:2] - self._cube0), 0.1))
        reward = r_reach + r_grasp + r_lift + r_success + r_pen
        self.info.grasped = self.info.grasped or grasped
        return reward, {"dist": dist, "grasped": grasped, "lift": lift, "r_reach": r_reach, "r_lift": r_lift}

    def _obs(self) -> np.ndarray:
        rb = self.robot
        q, qd = rb.q, rb.qd
        ee = rb.ee_pos()
        c = rb.body_pos(CUBE)
        cz_vel = rb.data.body(CUBE).cvel[5] if hasattr(rb.data.body(CUBE), "cvel") else 0.0
        obs = np.concatenate([
            q[:5], qd[:5] * 0.1, [rb.gripper_opening()], ee, c, c - ee, [float(cz_vel)], self._prev_action[[2, 3]],
            self._target - ee,  # where the commanded target sits relative to the fingertip
        ]).astype(np.float32)
        if self.dr.enabled and self.dr.obs_noise > 0:
            obs[11:20] += self.rng.normal(0, self.dr.obs_noise, 9).astype(np.float32)
        return obs

    def render(self):
        return self.robot.render(self.camera)

    def close(self):
        self.robot.close()


def make_env(seed: int = 0, dr: bool = True, **kw) -> SO101LiftEnv:
    return SO101LiftEnv(dr=DRConfig(enabled=dr), seed=seed, **kw)

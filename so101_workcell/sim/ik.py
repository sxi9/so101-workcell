"""Damped least-squares inverse kinematics for the SO-101 gripper site.

The SO-101 has 5 arm DOF, so arbitrary 6-DOF pose targets are unreachable. We solve for
position (3) plus the finger pointing direction (site x-axis, 2 effective DOF) and, with a
lower weight, the closing axis (site z-axis). That is enough for top-down grasps.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .robot import SO101


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    ori_err: float
    iters: int

    close_err: float = 0.0

    @property
    def ok(self) -> bool:
        return self.pos_err < 4e-3 and self.ori_err < 0.15 and self.close_err < 0.25


def _rotvec_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector that rotates unit vector a toward unit vector b."""
    return np.cross(a, b)


def solve_ik(
    robot: SO101,
    target_pos,
    finger_dir=(0.0, 0.0, -1.0),
    close_dir=None,
    q_init=None,
    *,
    max_iters: int = 300,
    damping: float = 1e-4,
    step: float = 1.0,
    w_close: float = 0.5,
    q_rest=None,
    w_rest: float = 0.1,
    close_dir_signed: bool = True,
) -> IKResult:
    """Solve for the 5 arm joints that put the fingertip site at target_pos.

    finger_dir: desired world direction of the site x-axis (where the fingers point).
    close_dir:  optional desired world direction of the site z-axis, which points from the
                fixed jaw toward the moving jaw. With close_dir_signed=False either sign is
                accepted (symmetric grasp); the default honours the sign so the caller can
                keep the moving jaw's sweep away from obstacles.
    """
    m = robot.model
    d = robot.scratch_data()
    d.qpos[:] = robot.data.qpos
    q = np.array(q_init if q_init is not None else robot.q[:5], dtype=float)
    lo, hi = robot.joint_low[:5], robot.joint_high[:5]
    q_rest = robot.q[:5] if q_rest is None else np.asarray(q_rest, float)
    dof = robot.qvel_idx[:5]
    target_pos = np.asarray(target_pos, float)
    fdir = np.asarray(finger_dir, float)
    fdir = fdir / np.linalg.norm(fdir)
    cdir = None
    if close_dir is not None:
        cdir = np.asarray(close_dir, float)
        cdir = cdir - fdir * (cdir @ fdir)  # must be orthogonal to the finger direction
        n = np.linalg.norm(cdir)
        cdir = cdir / n if n > 1e-6 else None
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    pos_err = ori_err = np.inf
    close_err = 0.0
    it = 0
    for it in range(max_iters):
        d.qpos[robot.qpos_idx[:5]] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = d.site_xpos[robot.site_id]
        R = d.site_xmat[robot.site_id].reshape(3, 3)
        e_pos = target_pos - p
        e_f = _rotvec_between(R[:, 0], fdir)
        rows = [e_pos, e_f]
        weights = [1.0, 1.0]
        if cdir is not None:
            cz = R[:, 2]
            c = cdir if (close_dir_signed or cz @ cdir >= 0) else -cdir
            rows.append(_rotvec_between(cz, c))
            weights.append(w_close)
        pos_err = float(np.linalg.norm(e_pos))
        ori_err = float(np.linalg.norm(e_f))
        close_err = float(np.linalg.norm(rows[2])) if len(rows) > 2 else 0.0
        if pos_err < 1e-4 and ori_err < 1e-3 and close_err < 1e-2:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, robot.site_id)
        J = np.vstack([jacp[:, dof]] + [jacr[:, dof]] * (len(rows) - 1))
        W = np.repeat(weights, 3)
        e = np.concatenate(rows) * W
        Jw = J * W[:, None]
        JJt = Jw @ Jw.T + damping * np.eye(Jw.shape[0])
        J_pinv = Jw.T @ np.linalg.inv(JJt)
        dq = J_pinv @ e
        # rest-posture pull projected into the task nullspace so it cannot bias the solution
        N = np.eye(len(q)) - J_pinv @ Jw
        dq += N @ (w_rest * (q_rest - q))
        q = np.clip(q + step * dq, lo, hi)
    return IKResult(q=q, pos_err=pos_err, ori_err=ori_err, iters=it + 1, close_err=close_err)

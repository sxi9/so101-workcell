"""Measure the SO-101 geometry numbers hard-coded in sim/skills.py and sim/robot.py.

Prints: gripperframe site axes, fingertip pad positions in the site frame, gripper gap vs
joint angle, and a fingers-down reachability map. Re-run after changing the MJCF.

python scripts/measure_geometry.py
"""
from __future__ import annotations

import numpy as np

from so101_workcell.sim.ik import solve_ik
from so101_workcell.sim.robot import SO101
from so101_workcell.sim.skills import REST_Q


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    r = SO101()
    r.set_q([0, 0, 0, 0, 0, 0])
    R = r.ee_rot()
    print("site axes at home: finger(x)", R[:, 0], " width(y)", R[:, 1], " close(z)", R[:, 2])

    print("\npad centres in the site frame (finger, width, close) and half-sizes:")
    for q in [0.0, 0.2, 0.5, 1.0]:
        r.set_q([0, 0, 0, 0, 0, q])
        R = r.ee_rot()
        for name in ["pad_fixed", "pad_moving"]:
            gid = r.model.geom(name).id
            rel = R.T @ (r.data.geom_xpos[gid] - r.ee_pos())
            print(f"  q={q:.1f} {name:10s} {rel}  size={r.model.geom_size[gid]}")

    print("\nreachability (fingers straight down), '#' = IK ok, '+' = <12 mm, '.' = unreachable")
    xs = np.linspace(0.10, 0.34, 13)
    ys = np.linspace(-0.22, 0.22, 12)
    for z in [0.02, 0.07]:
        print(f"  z={z:.2f}  y: {ys[0]:.2f} .. {ys[-1]:.2f}")
        for x in xs:
            row = ""
            for y in ys:
                rad = np.array([x, y, 0.0])
                rad /= np.linalg.norm(rad)
                tang = np.array([-rad[1], rad[0], 0.0])
                s = solve_ik(r, [x, y, z], (0, 0, -1), tang, q_init=REST_Q, q_rest=REST_Q)
                row += "#" if s.ok else ("+" if s.pos_err < 0.012 else ".")
            print(f"    x={x:.2f} {row}")


if __name__ == "__main__":
    main()

# Design notes

## Geometry that the code depends on

Measured with `scripts/measure_geometry.py` on the official SO-101 MJCF (new calibration).

| Quantity | Value | Where it is used |
|---|---|---|
| `gripperframe` site axes | x = finger direction, z = closing axis (fixed jaw -> moving jaw), y = jaw width | `sim/ik.py`, `sim/skills.py` |
| Fixed-pad inner face | 0.9 mm along +z from the site | `PAD_FACE_OFFSET` |
| Pad extent along the fingers | about -8 .. +10 mm from the site | `GRASP_HEIGHT_OFFSET` |
| Gripper gap vs joint angle | 8 mm at -0.17 rad, 21 mm at 0, 53 mm at 0.4, 96 mm at 1.0 | `GRIPPER_OPEN/CLOSED` |
| Fingers-down reach at table height | radius ~0.27 m; ~0.24 m at 7 cm height | spawn region, bin placement |
| Moving-jaw sweep while closing | 3..9 cm along +z, dipping to ~1 cm above the table | grasp-direction ranking |

## Things that went wrong and the fixes

1. **IK rest posture below the table.** The first rest pose had the fingertips at z = -9 mm. The
   solver now projects the rest pull into the task nullspace so it can never bias the solution.
2. **Pad landing on the cube.** A joint-space interpolation between hover and grasp does not
   stay vertical. `move_linear` solves IK on 1 cm waypoints so descents are straight.
3. **Sign-free closing axis.** The IK accepted either sign of the closing axis, but the pad
   geometry is attached to one of them. The sign is now honoured and read back from the
   achieved pose before the descent.
4. **Convex-hull finger collisions.** MuJoCo collides meshes by convex hull; the jaw hulls are
   much fatter than the printed parts and hit the cube before the pads did. The finger meshes
   are visual-only; two box pads are the only contact surfaces.
5. **Moving jaw sweeping into bins or neighbouring cubes.** The open jaw swings through a 9 cm
   arc that dips to table height. Grasp directions are ranked by the clearance of that arc.
6. **Contact time constants.** `solref` of 0.004 s is only twice the 0.002 s timestep and went
   unstable under domain randomization; 0.01 s is used everywhere.
7. **Control latency in the RL env.** A P-controller with gain 1 and a 1-step delay oscillates
   forever; the scripted oracle uses gain 0.5. Policies learn this themselves.
8. **Reward hacking by dragging.** With only a tanh reach term the gradient far from the cube
   is flat, and the first policy settled at a workspace corner, later learning to shove the cube
   there. A linear `-4 * dist` term, a penalty on cube displacement while not grasped, and the
   commanded target in the observation fixed it.
9. **Critic divergence under randomisation.** One physics blow-up sent the cube flying, the
   unbounded distance penalty produced a return of -2493, and the SAC critic loss grew to 1e21
   with the entropy coefficient following. The env now ends an episode on non-finite state or
   a cube off the table with a bounded penalty, clips per-step reward to +-12, and clips the
   observation. The nominal policy then transferred to randomised physics at 90% zero-shot.

## Sim -> twin -> real

Skills plan against local MuJoCo because grasp verification needs contact physics. The only
thing that leaves the planner is a 10 Hz stream of joint-position targets, which is what the
STS3215 servos accept. `sim/backends.py` sends that stream to local MuJoCo, to a Cyberwave
SO-101 twin (`playground`, `simulation`, or `live`), or to a dry-run logger, with no change to
the planning code. `tests/test_backends.py` proves a recorded stream reproduces the grasp in a
fresh simulator instance.

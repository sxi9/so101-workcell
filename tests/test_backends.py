import numpy as np

from so101_workcell.sim.backends import DryRunBackend, JointTrajectory, MujocoBackend
from so101_workcell.sim.robot import SO101
from so101_workcell.sim.scene import in_bin, randomize_cubes
from so101_workcell.sim.skills import CONTROL_HZ, Skills


def test_recorded_trajectory_replays_on_a_fresh_mujoco_instance():
    """Record joint targets while the skills run, then replay them open-loop into a second
    simulator with the same layout: the cube must still end up in the bin. This is the
    sim -> twin path: a Cyberwave twin or the real arm receives exactly this stream."""
    traj = JointTrajectory(hz=CONTROL_HZ)
    robot = SO101(seed=21)
    skills = Skills(robot, on_tick=traj.record_hook())
    randomize_cubes(robot)
    skills.home()
    assert skills.pick_and_place("cube_red", "bin_left").ok
    assert len(traj.targets) > 100

    twin = SO101(seed=21)  # same seed -> same layout
    randomize_cubes(twin)
    backend = MujocoBackend(twin)
    for q in traj.targets:
        backend.set_joint_targets(q)
        twin.step(int(round(1 / CONTROL_HZ / twin.dt)))
    assert in_bin(twin, "cube_red", "bin_left")


def test_dry_run_backend_and_save_load(tmp_path):
    traj = JointTrajectory(hz=10, targets=[np.arange(6.0) * i for i in range(5)])
    p = tmp_path / "t.npz"
    traj.save(p)
    loaded = JointTrajectory.load(p)
    assert loaded.hz == 10 and len(loaded.targets) == 5
    b = DryRunBackend()
    assert loaded.replay(b, realtime=False) == 5
    assert np.allclose(b.commands[-1], np.arange(6.0) * 4)

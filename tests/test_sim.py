import numpy as np
import pytest

from so101_workcell.sim.ik import solve_ik
from so101_workcell.sim.robot import SO101
from so101_workcell.sim.scene import CUBE_NAMES, in_bin, randomize_cubes
from so101_workcell.sim.skills import REST_Q, Skills


@pytest.fixture(scope="module")
def robot():
    return SO101(seed=0)


def test_model_loads(robot):
    assert robot.model.nu == 6
    assert {robot.model.camera(i).name for i in range(robot.model.ncam)} >= {"overhead", "front", "wrist"}
    for n in CUBE_NAMES + ["bin_left", "bin_right"]:
        robot.model.body(n)  # raises if missing


def test_ik_reaches_fingers_down_targets(robot):
    rng = np.random.default_rng(1)
    for _ in range(20):
        x, y, z = rng.uniform(0.14, 0.24), rng.uniform(-0.09, 0.09), rng.uniform(0.02, 0.07)
        res = solve_ik(robot, [x, y, z], (0, 0, -1), q_init=REST_Q, q_rest=REST_Q)
        assert res.ok, (x, y, z, res)
        p, R = robot.fk(res.q)
        assert np.linalg.norm(p - [x, y, z]) < 4e-3
        assert R[:, 0] @ np.array([0, 0, -1]) > np.cos(np.radians(10))


def test_randomize_keeps_cubes_apart_and_on_table(robot):
    state = randomize_cubes(robot, np.random.default_rng(3))
    ps = list(state.cubes.values())
    for i in range(len(ps)):
        assert 0.008 < ps[i][2] < 0.02
        for j in range(i + 1, len(ps)):
            assert np.linalg.norm(ps[i][:2] - ps[j][:2]) > 0.045


def test_scripted_pick_and_place_succeeds():
    robot = SO101(seed=7)
    skills = Skills(robot)
    randomize_cubes(robot)
    skills.home()
    r = skills.pick_and_place("cube_red", "bin_right")
    assert r.ok, r.reason
    assert in_bin(robot, "cube_red", "bin_right")

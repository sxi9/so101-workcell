import numpy as np

from so101_workcell.agent.executor import Executor
from so101_workcell.agent.perception import Detection, observe
from so101_workcell.agent.planner import RulePlanner
from so101_workcell.sim.robot import SO101
from so101_workcell.sim.scene import CUBE_NAMES, in_bin, randomize_cubes
from so101_workcell.sim.skills import Skills


def test_perception_localizes_cubes_within_1cm():
    robot = SO101(seed=5)
    randomize_cubes(robot)
    Skills(robot).home()
    _img, dets = observe(robot)
    assert {d.color for d in dets} == {"red", "green", "blue"}
    for d in dets:
        truth = robot.body_pos(f"cube_{d.color}")[:2]
        err = np.linalg.norm(np.array(d.table_xy) - truth)
        assert err < 0.01, (d.color, err)


def _dets(*colors):
    return [Detection(c, (0, 0), (0.2, 0.0), 50) for c in colors]


def test_rule_planner_parses_common_instructions():
    p = RulePlanner()
    plan = p.plan("Put the red cube in the left bin", _dets("red", "green", "blue"))
    assert [(s.object, s.target) for s in plan.steps] == [("red", "bin_left")]
    plan = p.plan("move every cube to the right bin", _dets("red", "green"))
    assert sorted((s.object, s.target) for s in plan.steps) == [("green", "bin_right"), ("red", "bin_right")]
    plan = p.plan("red and blue go left, the others go right", _dets("red", "green", "blue"))
    assert sorted((s.object, s.target) for s in plan.steps) == [("blue", "bin_left"), ("green", "bin_right"), ("red", "bin_left")]
    plan = p.plan("put the yellow cube in the left bin", _dets("red"))
    assert not plan.feasible


def test_executor_runs_plan_end_to_end():
    robot = SO101(seed=11)
    skills = Skills(robot)
    randomize_cubes(robot)
    skills.home()
    _img, dets = observe(robot)
    plan = RulePlanner().plan("red to the right bin, everything else to the left bin", dets)
    log = Executor(robot, skills).run("test", plan, "rule")
    assert log.ok, [s.reason for s in log.steps]
    assert in_bin(robot, "cube_red", "bin_right")
    for n in CUBE_NAMES:
        if n != "cube_red":
            assert in_bin(robot, n, "bin_left")

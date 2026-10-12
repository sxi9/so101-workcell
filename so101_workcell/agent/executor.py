"""Plan executor: runs skill steps with verification and bounded retries, logs everything."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from so101_workcell.sim.robot import SO101
from so101_workcell.sim.scene import in_bin
from so101_workcell.sim.skills import Skills

from .planner import Plan, Step


@dataclass
class StepLog:
    step: dict
    attempts: int
    ok: bool
    reason: str
    seconds: float


@dataclass
class ExecutionLog:
    instruction: str
    planner: str
    plan: dict
    steps: list[StepLog] = field(default_factory=list)
    ok: bool = False
    sim_seconds: float = 0.0
    wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Executor:
    def __init__(self, robot: SO101, skills: Skills, *, max_attempts: int = 3):
        self.robot = robot
        self.skills = skills
        self.max_attempts = max_attempts

    def run_step(self, step: Step) -> StepLog:
        t0 = time.time()
        reason = ""
        for attempt in range(1, self.max_attempts + 1):
            if in_bin(self.robot, step.cube, step.target):
                return StepLog(asdict(step), attempt - 1, True, "already in target bin", time.time() - t0)
            r = self.skills.pick_and_place(step.cube, step.target)
            reason = r.reason
            if r.ok:
                return StepLog(asdict(step), attempt, True, reason, time.time() - t0)
            # recovery: open the gripper and back off before trying again
            self.skills.robot.set_gripper(1.0)
            self.skills.home()
        return StepLog(asdict(step), self.max_attempts, False, reason, time.time() - t0)

    def run(self, instruction: str, plan: Plan, planner_name: str) -> ExecutionLog:
        log = ExecutionLog(instruction, planner_name, {"steps": [asdict(s) for s in plan.steps], "reasoning": plan.reasoning, "infeasible_reason": plan.infeasible_reason})
        t0 = time.time()
        if not plan.feasible:
            log.ok = False
            log.wall_seconds = time.time() - t0
            return log
        self.skills.home()
        for step in plan.steps:
            sl = self.run_step(step)
            log.steps.append(sl)
            print(f"  {step.object:6s} -> {step.target:9s} : {'OK ' if sl.ok else 'FAIL'} after {sl.attempts} attempt(s) ({sl.reason})")
        self.skills.home()
        log.ok = all(s.ok for s in log.steps)
        log.sim_seconds = self.robot.time
        log.wall_seconds = time.time() - t0
        return log

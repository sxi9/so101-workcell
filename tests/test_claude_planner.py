"""ClaudePlanner is exercised against a stubbed client: no network, no API key."""
import json
from types import SimpleNamespace

import numpy as np

from so101_workcell.agent.perception import Detection
from so101_workcell.agent.planner import PLAN_SCHEMA, ClaudePlanner


class _StubMessages:
    def __init__(self, payload: dict, stop_reason: str = "end_turn"):
        self.payload = payload
        self.stop_reason = stop_reason
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="text", text=json.dumps(self.payload))
        details = SimpleNamespace(category="cyber") if self.stop_reason == "refusal" else None
        return SimpleNamespace(content=[block], stop_reason=self.stop_reason, stop_details=details)


def _planner(payload, stop_reason="end_turn"):
    p = ClaudePlanner.__new__(ClaudePlanner)
    p.model = "claude-opus-5-5"
    p.effort = "medium"
    stub = _StubMessages(payload, stop_reason)
    p.client = SimpleNamespace(beta=SimpleNamespace(messages=stub))
    return p, stub


def test_claude_planner_parses_schema_constrained_json():
    payload = {"reasoning": "red goes left", "steps": [{"action": "pick_and_place", "object": "red", "target": "bin_left"}], "infeasible_reason": ""}
    p, stub = _planner(payload)
    dets = [Detection("red", (10, 10), (0.2, 0.0), 200)]
    plan = p.plan("put the red cube in the left bin", dets, np.zeros((8, 8, 3), np.uint8))
    assert plan.feasible and plan.steps[0].cube == "cube_red" and plan.steps[0].target == "bin_left"
    call = stub.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["schema"] == PLAN_SCHEMA
    assert call["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in call["betas"]
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "image" and "Detected objects" in content[1]["text"]


def test_claude_planner_handles_refusal_and_infeasible():
    p, _ = _planner({"reasoning": "", "steps": [], "infeasible_reason": ""}, stop_reason="refusal")
    plan = p.plan("x", [], None)
    assert not plan.feasible and "refusal" in plan.infeasible_reason
    p, _ = _planner({"reasoning": "no yellow", "steps": [], "infeasible_reason": "no yellow cube visible"})
    plan = p.plan("move the yellow cube", [], None)
    assert not plan.feasible and plan.infeasible_reason == "no yellow cube visible"

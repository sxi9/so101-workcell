"""Task planners: turn an instruction plus a scene observation into a list of skill calls.

"The model decides WHAT, the code decides HOW." A plan is a short list of
(action, object, target) steps that the executor runs with verified, retrying skills. Two
planners share the interface:

* RulePlanner  - offline keyword parser, deterministic, used in tests and as a fallback.
* ClaudePlanner - vision-language planner on the Claude API. It sees the overhead image and
  the detector's object list and returns a JSON plan constrained by a schema.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .perception import Detection, scene_description

COLORS = ["red", "green", "blue", "yellow"]
BINS = {"bin_left": ["left", "blue tray", "blue bin"], "bin_right": ["right", "orange tray", "orange bin", "red tray"]}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "One or two sentences on how the instruction maps to the detected objects."},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["pick_and_place"]},
                    "object": {"type": "string", "enum": COLORS, "description": "colour of the cube to move"},
                    "target": {"type": "string", "enum": ["bin_left", "bin_right"]},
                },
                "required": ["action", "object", "target"],
                "additionalProperties": False,
            },
        },
        "infeasible_reason": {"type": "string", "description": "Empty string if the plan is feasible; otherwise why the instruction cannot be carried out with the detected objects."},
    },
    "required": ["reasoning", "steps", "infeasible_reason"],
    "additionalProperties": False,
}


@dataclass
class Step:
    action: str
    object: str  # colour
    target: str  # bin name

    @property
    def cube(self) -> str:
        return f"cube_{self.object}"


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    reasoning: str = ""
    infeasible_reason: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return not self.infeasible_reason and bool(self.steps)


class RulePlanner:
    """Keyword parser. Handles 'put the red cube in the left bin', 'move every cube to the
    right bin', 'red and blue left, green right', 'sort: red left, others right'."""

    name = "rule"

    def plan(self, instruction: str, dets: list[Detection], image: np.ndarray | None = None) -> Plan:
        text = instruction.lower()
        present = [d.color for d in dets]
        clauses = re.split(r"[;,.]|\bthen\b|\band\s+(?=the|put|move|place|every|all)", text)
        steps: list[Step] = []
        assigned: set[str] = set()
        for clause in clauses:
            if not clause.strip():
                continue
            target = self._bin(clause)
            colors = [c for c in COLORS if re.search(rf"\b{c}\b", clause)]
            wants_all = bool(re.search(r"\b(all|every|each|everything)\b", clause))
            wants_rest = bool(re.search(r"\b(others?|rest|remaining)\b", clause))
            if target is None:
                continue
            if wants_all:
                colors = [c for c in present if c not in assigned]
            elif wants_rest:
                colors = [c for c in present if c not in assigned]
            for c in colors:
                if c in assigned:
                    continue
                if c not in present:
                    return Plan(reasoning=f"'{c}' cube requested but not detected", infeasible_reason=f"no {c} cube visible")
                steps.append(Step("pick_and_place", c, target))
                assigned.add(c)
        if not steps:
            return Plan(reasoning="could not parse instruction", infeasible_reason="instruction not understood by the rule planner")
        return Plan(steps=steps, reasoning="keyword match")

    @staticmethod
    def _bin(clause: str) -> str | None:
        for name, words in BINS.items():
            if any(w in clause for w in words):
                return name
        return None


class ClaudePlanner:
    """Vision-language planner on the Claude API (Opus 5.5 by default).

    Sends the overhead image and the detector's object list; the response is constrained to
    PLAN_SCHEMA so the executor can trust the field names. Server-side refusal fallbacks are
    enabled by default so a safeguard decline on the primary model re-runs on a fallback.
    """

    name = "claude"
    SYSTEM = (
        "You plan pick-and-place tasks for a small desktop robot arm (SO-101) that can only move "
        "coloured cubes into one of two bins. You are given an overhead camera image and a list "
        "of detected objects with table coordinates. Map the user's instruction onto those objects "
        "and output the minimal ordered list of pick_and_place steps. Only use colours that were "
        "detected. If the instruction asks for something the arm cannot do with the detected "
        "objects, return no steps and explain in infeasible_reason."
    )

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium"):
        import anthropic  # optional dependency: pip install "so101-workcell[vlm]"

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def plan(self, instruction: str, dets: list[Detection], image: np.ndarray | None = None) -> Plan:
        content: list[dict] = []
        if image is not None:
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="PNG")
            data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        content.append({"type": "text", "text": f"{scene_description(dets)}\n\nInstruction: {instruction}"})
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            cat = response.stop_details.category if response.stop_details else None
            return Plan(reasoning="model declined", infeasible_reason=f"planner refusal ({cat})")
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        steps = [Step(s["action"], s["object"], s["target"]) for s in data.get("steps", [])]
        return Plan(steps=steps, reasoning=data.get("reasoning", ""), infeasible_reason=data.get("infeasible_reason", ""), raw=data)


def make_planner(name: str) -> RulePlanner | ClaudePlanner:
    if name == "rule":
        return RulePlanner()
    if name == "claude":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # the SDK also reads `ant auth login` profiles; let it try and fail with a clear error
            pass
        return ClaudePlanner()
    raise ValueError(f"unknown planner {name!r}")

"""PR review simulation environment (gym-style reset/step API)."""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Optional

from .grader import check_comment, grade
from .models import PRReviewAction, PRReviewObservation, PRReviewReward

_BUG_POOL = 0.7       # reward split evenly across all bugs in a scenario
_FALSE_POS = -0.05    # penalty per comment that identifies no real bug
_DECISION_CORRECT = 0.3
_DECISION_WRONG = -0.3

_SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "scenarios")

TASK_PREFIXES = {"easy": "easy_", "medium": "medium_", "hard": "hard_"}
TASK_MAX_STEPS = {"easy": 5, "medium": 10, "hard": 15}
TASK_THRESHOLDS = {"easy": 0.7, "medium": 0.6, "hard": 0.5}


def _load_all() -> dict[str, dict]:
    paths = glob.glob(os.path.join(_SCENARIOS_DIR, "*.json"))
    if not paths:
        raise RuntimeError(f"No scenario JSON files found in {_SCENARIOS_DIR}")
    store: dict[str, dict] = {}
    for path in sorted(paths):
        sid = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for field in ("pr_title", "pr_description", "diff", "ground_truth"):
            if field not in data:
                raise ValueError(f"Scenario '{sid}' missing field '{field}'")
        store[sid] = data
    return store


_STORE: dict[str, dict] = _load_all()


class PRReviewEnv:
    def __init__(self, task: str = "easy") -> None:
        if task not in TASK_PREFIXES:
            raise ValueError(f"Unknown task '{task}'. Valid: {sorted(TASK_PREFIXES)}")
        self.task = task
        self.max_steps: int = TASK_MAX_STEPS[task]
        self.threshold: float = TASK_THRESHOLDS[task]
        self._scenario_id: Optional[str] = None
        self._scenario: Optional[dict] = None
        self._comments: list[str] = []
        self._step_count: int = 0
        self._done: bool = False
        self._score: Optional[float] = None
        self._rewarded_bugs: set[int] = set()

    def reset(self) -> PRReviewObservation:
        prefix = TASK_PREFIXES[self.task]
        candidates = [sid for sid in _STORE if sid.startswith(prefix)]
        if not candidates:
            raise RuntimeError(f"No scenarios with prefix '{prefix}'")
        self._scenario_id = random.choice(candidates)
        self._scenario = _STORE[self._scenario_id]
        self._comments = []
        self._step_count = 0
        self._done = False
        self._score = None
        self._rewarded_bugs = set()
        return self._obs()

    def step(self, action: PRReviewAction) -> tuple[PRReviewObservation, PRReviewReward, bool, dict]:
        if self._scenario is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode done. Call reset() to start a new one.")
        if self._step_count >= self.max_steps:
            return self._terminal_step("reject")

        self._step_count += 1

        if action.action_type == "comment":
            reward_val = self._comment_reward(action.body)
            if action.body:
                self._comments.append(action.body)
            clipped = round(max(0.01, min(0.99, reward_val)), 4)
            return self._obs(), PRReviewReward(value=clipped), False, {}

        if action.action_type in ("approve", "request_changes"):
            decision = "approve" if action.action_type == "approve" else "reject"
            return self._terminal_step(decision)

        raise ValueError(f"Unknown action_type '{action.action_type}'.")

    def state(self) -> dict:
        return {
            "task": self.task,
            "scenario_id": self._scenario_id,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "done": self._done,
            "score": self._score,
            "comments": list(self._comments),
        }

    def _obs(self) -> PRReviewObservation:
        assert self._scenario is not None
        return PRReviewObservation(
            diff=self._scenario["diff"],
            pr_description=self._scenario["pr_description"],
            pr_title=self._scenario["pr_title"],
            comments_so_far=[{"body": c} for c in self._comments],
            step_count=self._step_count,
            done=self._done,
            scenario_id=self._scenario_id or "",
        )

    def _comment_reward(self, body: str) -> float:
        if not body:
            return 0.0
        assert self._scenario is not None
        bugs: list = self._scenario["ground_truth"].get("bugs", [])
        if not bugs:
            return _FALSE_POS  # any comment on a clean PR is hallucination
        newly_found = [i for i in check_comment(body, bugs) if i not in self._rewarded_bugs]
        if newly_found:
            per_bug = _BUG_POOL / len(bugs)
            self._rewarded_bugs.update(newly_found)
            return len(newly_found) * per_bug
        return _FALSE_POS

    def _terminal_step(self, decision: str) -> tuple[PRReviewObservation, PRReviewReward, bool, dict]:
        assert self._scenario is not None
        result = grade(
            ground_truth=self._scenario["ground_truth"],
            comments=self._comments,
            decision=decision,
        )
        self._done = True
        self._score = round(max(0.01, min(0.99, result["score"])), 4)
        result["score"] = self._score
        decision_reward = _DECISION_CORRECT if result["decision_correct"] else _DECISION_WRONG
        clipped_reward = round(max(0.01, min(0.99, decision_reward)), 4)
        return self._obs(), PRReviewReward(value=clipped_reward, breakdown=result), True, result

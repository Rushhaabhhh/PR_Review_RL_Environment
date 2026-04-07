"""
PR Review OpenEnv — FastAPI server.

Endpoints:
  POST /reset          Load a scenario (random or by ID).
  POST /step           Submit a comment or approve/reject decision.
  GET  /state          Inspect current session state.
  GET  /scenarios      List all available scenario IDs.
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Make sure src/ is importable when running as `uvicorn src.api:app`
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from grader import grade  # noqa: E402

# ---------------------------------------------------------------------------
# Startup: load every scenario file into memory once.
# ---------------------------------------------------------------------------
_SCENARIOS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "scenarios"
)

# Maps scenario_id -> full parsed JSON (including ground_truth, kept private)
_scenario_store: dict[str, dict] = {}


def _load_scenarios() -> None:
    paths = glob.glob(os.path.join(_SCENARIOS_DIR, "*.json"))
    if not paths:
        raise RuntimeError(f"No scenario JSON files found in {_SCENARIOS_DIR}")
    for path in sorted(paths):
        sid = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Validate required fields at startup so we fail loud, not mid-run.
        for field in ("pr_title", "pr_description", "diff", "ground_truth"):
            if field not in data:
                raise ValueError(f"Scenario {sid} is missing field '{field}'")
        gt = data["ground_truth"]
        for field in ("bugs", "should_approve"):
            if field not in gt:
                raise ValueError(
                    f"Scenario {sid}.ground_truth is missing field '{field}'"
                )
        _scenario_store[sid] = data


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _load_scenarios()
    yield


app = FastAPI(title="PR Review OpenEnv", version="1.0.0", lifespan=_lifespan)

# ---------------------------------------------------------------------------
# In-memory session state (single session; reset blows it away)
# ---------------------------------------------------------------------------
_state: dict = {
    "status": "idle",       # "idle" | "reviewing" | "done"
    "scenario_id": None,
    "scenario": None,       # full dict — never sent to client
    "comments": [],
    "decision": None,       # "approve" | "reject"
    "score": None,          # grader output dict
}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    scenario_id: Optional[str] = None
    force: bool = False  # set True to reset mid-session without error


class StepRequest(BaseModel):
    action: str           # "comment" | "approve" | "reject"
    content: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _public_state() -> dict:
    """Return the session state safe to send to the client (no ground_truth)."""
    return {
        "status": _state["status"],
        "scenario_id": _state["scenario_id"],
        "comments": list(_state["comments"]),
        "decision": _state["decision"],
        "score": _state["score"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/scenarios")
def list_scenarios() -> dict:
    """Return all available scenario IDs."""
    return {"scenarios": sorted(_scenario_store.keys())}


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()) -> dict:
    """
    Start a new review session.

    If scenario_id is omitted a random scenario is chosen.
    Returns the PR title, description, and diff — but NOT ground_truth.
    """
    global _state

    if _state["status"] == "reviewing" and not body.force:
        raise HTTPException(
            status_code=409,
            detail=(
                "A session is already in progress. "
                "Submit a decision or pass force=true to reset."
            ),
        )

    if body.scenario_id:
        if body.scenario_id not in _scenario_store:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown scenario_id '{body.scenario_id}'. "
                       f"Available: {sorted(_scenario_store.keys())}",
            )
        sid = body.scenario_id
    else:
        sid = random.choice(list(_scenario_store.keys()))

    scenario = _scenario_store[sid]

    _state = {
        "status": "reviewing",
        "scenario_id": sid,
        "scenario": scenario,
        "comments": [],
        "decision": None,
        "score": None,
    }

    return {
        "status": "reviewing",
        "scenario_id": sid,
        "pr_title": scenario["pr_title"],
        "pr_description": scenario["pr_description"],
        "diff": scenario["diff"],
    }


@app.post("/step")
def step(body: StepRequest) -> dict:
    """
    Advance the session.

    action="comment"  — record a review comment (content required).
    action="approve"  — approve the PR and end the session.
    action="reject"   — reject the PR and end the session.
    """
    global _state

    if _state["status"] == "idle":
        raise HTTPException(status_code=400, detail="Call /reset first.")

    if _state["status"] == "done":
        raise HTTPException(
            status_code=400,
            detail="Session is done. Call /reset to start a new episode.",
        )

    if body.action == "comment":
        if not body.content or not body.content.strip():
            raise HTTPException(
                status_code=422, detail="action='comment' requires non-empty content."
            )
        _state["comments"].append(body.content.strip())
        return {
            "status": "reviewing",
            "comments_recorded": len(_state["comments"]),
        }

    elif body.action in ("approve", "reject"):
        _state["decision"] = body.action
        _state["status"] = "done"
        result = grade(
            ground_truth=_state["scenario"]["ground_truth"],
            comments=_state["comments"],
            decision=body.action,
        )
        _state["score"] = result
        return {
            "status": "done",
            "decision": body.action,
            **result,
        }

    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown action '{body.action}'. Use 'comment', 'approve', or 'reject'.",
        )


@app.get("/state")
def state() -> dict:
    """Return the current session state (no ground_truth exposed)."""
    return _public_state()

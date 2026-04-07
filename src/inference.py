"""
PR Review Agent — calls the OpenEnv API, uses an LLM to review diffs.

Environment variables:
  API_BASE_URL   Base URL for OpenAI-compatible LLM endpoint (required)
  API_KEY        Auth key for the LLM API (required)
  MODEL          Model name to use (default: claude-opus-4-6)
  ENV_URL        PR Review OpenEnv server URL (default: http://localhost:8000)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — fail loudly if required env vars are missing
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "")
API_KEY: str = os.environ.get("API_KEY", "")
MODEL: str = os.environ.get("MODEL", "claude-opus-4-6")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:8000")

if not API_BASE_URL:
    sys.exit(
        "ERROR: API_BASE_URL is required. "
        "Set it in your environment or docker-compose.yml."
    )
if not API_KEY:
    sys.exit(
        "ERROR: API_KEY is required. "
        "Set it in your environment or docker-compose.yml."
    )

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a senior software engineer performing a thorough code review.

Your task:
1. Read the pull request title, description, and unified diff carefully.
2. Identify ALL bugs, security vulnerabilities, and logic errors — be specific.
3. For each issue state: what it is, why it is dangerous, and how to fix it.
4. Decide whether to approve or reject the PR.

Rules:
- Reject if there are any bugs, security issues, or correctness problems.
- Approve only if the code is clean, correct, and safe.
- Do not hallucinate bugs that are not in the diff.

Respond with valid JSON only — no markdown fences, no extra text:
{
  "comments": [
    "Issue 1: <description of bug, why dangerous, how to fix>",
    "Issue 2: ..."
  ],
  "decision": "approve" | "reject",
  "reasoning": "<one sentence explaining the overall decision>"
}
"""


def _call_llm(pr_title: str, pr_description: str, diff: str) -> dict:
    """Call the LLM and parse the JSON review response."""
    user_message = (
        f"## Pull Request: {pr_title}\n\n"
        f"### Description\n{pr_description}\n\n"
        f"### Diff\n```diff\n{diff}\n```\n\n"
        "Review the diff and respond with JSON as instructed."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""

    # Strip accidental markdown code fences if the model adds them
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Best-effort: extract the first JSON object from the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Fallback: treat the whole response as a single comment, reject
        print(f"[warn] Could not parse LLM response as JSON:\n{raw[:300]}")
        return {
            "comments": [raw],
            "decision": "reject",
            "reasoning": "Unparseable response — defaulting to reject.",
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(scenario_id: Optional[str] = None) -> dict:
    """
    Run one full review episode.

    Returns the final score dict from the environment.
    """
    # 1. Reset environment — always force=True so a previously crashed episode
    #    (which may have reset but never submitted a decision) never blocks this run.
    reset_payload: dict = {"force": True}
    if scenario_id:
        reset_payload["scenario_id"] = scenario_id

    resp = requests.post(f"{ENV_URL}/reset", json=reset_payload, timeout=10)
    resp.raise_for_status()
    env = resp.json()

    sid = env["scenario_id"]
    print(f"\n{'='*60}")
    print(f"Scenario : {sid}")
    print(f"PR Title : {env['pr_title']}")
    print(f"{'='*60}")

    # 2. Ask LLM to review
    review = _call_llm(env["pr_title"], env["pr_description"], env["diff"])
    comments: list[str] = review.get("comments", [])
    decision: str = review.get("decision", "reject")

    # 3. Submit comments
    for i, comment in enumerate(comments, 1):
        resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": "comment", "content": comment},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"Comment {i}: {comment[:120]}{'...' if len(comment) > 120 else ''}")

    # 4. Submit decision
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": decision},
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json()

    # 5. Print summary
    print(f"\nDecision          : {decision}")
    print(f"Decision correct  : {result.get('decision_correct')}")
    print(f"Bugs found        : {result.get('bugs_found')}/{result.get('total_bugs')}")
    print(f"Bug detection rate: {result.get('bug_detection_rate'):.1%}")
    print(f"Final score       : {result.get('score'):.4f}")

    return result


def run_all_scenarios() -> None:
    """Run every scenario in order and print an aggregate summary."""
    resp = requests.get(f"{ENV_URL}/scenarios", timeout=10)
    resp.raise_for_status()
    scenario_ids: list[str] = resp.json()["scenarios"]

    results = []
    for sid in scenario_ids:
        try:
            result = run_episode(sid)
            results.append((sid, result["score"]))
        except Exception as exc:
            print(f"[error] {sid}: {exc}")
            results.append((sid, 0.0))

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    for sid, score in results:
        print(f"  {sid:<40} {score:.4f}")
    avg = sum(s for _, s in results) / len(results) if results else 0.0
    print(f"\n  Average score: {avg:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No argument → run all scenarios
        run_all_scenarios()
    elif sys.argv[1] == "--all":
        run_all_scenarios()
    else:
        # Single scenario ID provided
        run_episode(sys.argv[1])

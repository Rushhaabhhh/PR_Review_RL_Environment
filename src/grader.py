"""Pure grading functions — no I/O, no global state."""

from __future__ import annotations
import re


def _keyword_found(keyword: str, text: str) -> bool:
    """Case-insensitive search. Uses word boundaries for alphanumeric keywords
    to avoid substring false positives (e.g. 'null' matching 'nullable')."""
    kw = keyword.lower()
    text = text.lower()
    if kw and re.match(r"\w", kw[0]) and re.match(r"\w", kw[-1]):
        return bool(re.search(r"\b" + re.escape(kw) + r"\b", text))
    return kw in text


def check_comment(comment: str, bugs: list) -> list[int]:
    """Return indices of bugs matched by this comment (for step-level rewards)."""
    text = comment.lower()
    matched: list[int] = []
    for i, keyword_list in enumerate(bugs):
        if isinstance(keyword_list, str):
            keyword_list = [keyword_list]
        if any(_keyword_found(kw, text) for kw in keyword_list):
            matched.append(i)
    return matched


def grade(ground_truth: dict, comments: list[str], decision: str) -> dict:
    """Score a completed review session against ground truth.

    Returns score in [0, 1] = bug_detection * 0.7 + decision * 0.3,
    minus 0.2 false-rejection penalty if agent rejects a clean PR.
    """
    full_text = " ".join(comments).lower()
    bugs: list = ground_truth.get("bugs", [])
    should_approve: bool = ground_truth.get("should_approve", False)

    bug_breakdown = []
    bugs_found = 0
    for keyword_list in bugs:
        if isinstance(keyword_list, str):
            keyword_list = [keyword_list]
        matched_kw = next((kw for kw in keyword_list if _keyword_found(kw, full_text)), None)
        found = matched_kw is not None
        if found:
            bugs_found += 1
        bug_breakdown.append({"keywords": keyword_list, "found": found, "matched_by": matched_kw})

    total_bugs = len(bugs)
    bug_detection_rate = bugs_found / total_bugs if total_bugs > 0 else 1.0
    decision_correct = (decision == "approve") == should_approve
    decision_score = 1.0 if decision_correct else 0.0

    false_rejection = should_approve and decision == "reject"
    false_rejection_penalty = -0.2 if false_rejection else 0.0

    raw_score = bug_detection_rate * 0.7 + decision_score * 0.3 + false_rejection_penalty
    final_score = round(max(0.01, min(0.99, raw_score)), 4)

    return {
        "score": final_score,
        "bug_detection_rate": round(bug_detection_rate, 4),
        "bugs_found": bugs_found,
        "total_bugs": total_bugs,
        "decision_correct": decision_correct,
        "decision_score": decision_score,
        "false_rejection_penalty": false_rejection_penalty,
        "bug_breakdown": bug_breakdown,
    }

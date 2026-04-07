"""
Grader: pure function that scores an agent's PR review against ground truth.
No I/O, no global state — safe to import anywhere.
"""

from __future__ import annotations
import re


def _keyword_found(keyword: str, text: str) -> bool:
    """
    Case-insensitive keyword search.

    - For keywords that start AND end with a word character (letter/digit/underscore),
      use \\b boundaries to avoid substring false positives (e.g. "null" in "nullable").
    - For keywords that contain punctuation or special characters, use plain substring
      matching since \\b is undefined at non-word characters.
    """
    kw = keyword.lower()
    # Check whether the keyword boundaries are word characters
    leading_word = bool(re.match(r"\w", kw[0])) if kw else False
    trailing_word = bool(re.match(r"\w", kw[-1])) if kw else False

    if leading_word and trailing_word:
        pattern = r"\b" + re.escape(kw) + r"\b"
        return bool(re.search(pattern, text))
    else:
        return kw in text


def grade(
    ground_truth: dict,
    comments: list[str],
    decision: str,
) -> dict:
    """
    Score a review session.

    Parameters
    ----------
    ground_truth : dict
        {"bugs": [[kw, ...], ...], "should_approve": bool}
        Each inner list is a set of alternative keywords for ONE bug.
        A bug is detected if ANY keyword from its list appears in the comments.
    comments : list[str]
        All comment strings submitted by the agent.
    decision : str
        "approve" or "reject"

    Returns
    -------
    dict with keys: score, bug_detection_rate, bugs_found, total_bugs,
                    decision_correct, bug_breakdown
    """
    full_text = " ".join(comments).lower()

    bugs: list[list[str]] = ground_truth.get("bugs", [])
    should_approve: bool = ground_truth.get("should_approve", False)

    bug_breakdown = []
    bugs_found = 0

    for keyword_list in bugs:
        # Normalise: accept a bare string or a list of strings
        if isinstance(keyword_list, str):
            keyword_list = [keyword_list]
        matched_kw = next(
            (kw for kw in keyword_list if _keyword_found(kw, full_text)),
            None,
        )
        found = matched_kw is not None
        if found:
            bugs_found += 1
        bug_breakdown.append(
            {
                "keywords": keyword_list,
                "found": found,
                "matched_by": matched_kw,
            }
        )

    total_bugs = len(bugs)
    bug_detection_rate = bugs_found / total_bugs if total_bugs > 0 else 1.0

    decision_correct = (decision == "approve") == should_approve
    decision_score = 1.0 if decision_correct else 0.0

    final_score = round(bug_detection_rate * 0.7 + decision_score * 0.3, 4)

    return {
        "score": final_score,
        "bug_detection_rate": round(bug_detection_rate, 4),
        "bugs_found": bugs_found,
        "total_bugs": total_bugs,
        "decision_correct": decision_correct,
        "decision_score": decision_score,
        "bug_breakdown": bug_breakdown,
    }

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any

from .types import JudgeResult, RubricItem, ValidationFinding


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge response did not contain a JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Judge response JSON must be an object")
    return value


def normalize_judge_result(
    judge_name: str,
    candidate_id: str,
    payload: dict[str, Any],
    rubric: tuple[RubricItem, ...],
    latency_seconds: float | None = None,
    estimated_cost_usd: float | None = None,
) -> JudgeResult:
    raw_scores = payload.get("scores", {})
    scores: dict[str, float] = {}
    for item in rubric:
        raw_value = float(raw_scores.get(item.id, 0.0))
        scores[item.id] = max(0.0, min(raw_value, item.max_points))
    return JudgeResult(
        judge=judge_name,
        candidate_id=candidate_id,
        scores=scores,
        fatal_errors=[str(value) for value in payload.get("fatal_errors", [])],
        summary=str(payload.get("summary", "")),
        latency_seconds=latency_seconds,
        estimated_cost_usd=estimated_cost_usd,
    )


def median_quality(judgments: list[JudgeResult]) -> float:
    if not judgments:
        return 0.0
    return float(statistics.median(result.total for result in judgments))


def deterministic_penalty(findings: list[ValidationFinding]) -> float:
    penalty = 0.0
    for finding in findings:
        if finding.passed:
            continue
        penalty += 10.0 if finding.severity == "error" else 3.0
    return penalty


def adjusted_quality(judgments: list[JudgeResult], findings: list[ValidationFinding]) -> float:
    return max(0.0, median_quality(judgments) - deterministic_penalty(findings))


def disagreement_index(judgments: list[JudgeResult], max_score: float = 100.0) -> float:
    if len(judgments) < 2 or max_score <= 0:
        return 0.0
    deviation = statistics.pstdev(result.total for result in judgments)
    return min(1.0, deviation / (max_score / 2.0))


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is",
        "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "will", "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.casefold())
        if len(token) > 1 and token not in stopwords
    }


def lexical_convergence(texts: list[str]) -> float:
    """Mean pairwise Jaccard overlap, used only as one bounded convergence signal."""
    if len(texts) < 2:
        return 1.0
    token_sets = [_content_tokens(text) for text in texts]
    similarities: list[float] = []
    for left_index, left in enumerate(token_sets):
        for right in token_sets[left_index + 1 :]:
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 1.0)
    return statistics.fmean(similarities) if similarities else 1.0


def hard_constraints_pass(findings: list[ValidationFinding]) -> bool:
    return all(finding.passed for finding in findings if finding.severity == "error")


def efficiency_score(
    latency_seconds: float | None,
    cost_usd: float | None,
    latency_target: float = 20.0,
    cost_target: float = 0.10,
) -> float:
    latency_component = 50.0 if latency_seconds is None else 100.0 / (1.0 + latency_seconds / latency_target)
    cost_component = 50.0 if cost_usd is None else 100.0 / (1.0 + cost_usd / cost_target)
    return 0.5 * latency_component + 0.5 * cost_component


def decision_utility(quality: float, efficiency: float, quality_weight: float = 0.9) -> float:
    quality_weight = max(0.0, min(quality_weight, 1.0))
    return quality_weight * quality + (1.0 - quality_weight) * efficiency


def confidence_from_quality_and_disagreement(quality: float, disagreement: float) -> float:
    quality_fraction = max(0.0, min(quality / 100.0, 1.0))
    return quality_fraction * (1.0 - 0.5 * max(0.0, min(disagreement, 1.0)))


def score_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }

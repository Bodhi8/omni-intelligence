from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .types import Task, ValidationFinding


_CURRENCY = re.compile(r"\$\s*\d[\d,]*(?:\.\d+)?\s*[KkMm]?")


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def parse_currency(value: str) -> float:
    cleaned = value.replace("$", "").replace(",", "").replace(" ", "")
    multiplier = 1.0
    if cleaned[-1:].lower() == "k":
        multiplier = 1_000.0
        cleaned = cleaned[:-1]
    elif cleaned[-1:].lower() == "m":
        multiplier = 1_000_000.0
        cleaned = cleaned[:-1]
    return float(cleaned) * multiplier


def _allocation_segment(text: str, label: str, labels: Iterable[str]) -> str | None:
    for line in text.splitlines():
        position = line.casefold().find(label.casefold())
        if position < 0:
            continue
        segment = line[position + len(label) :]
        boundaries = []
        for other_label in labels:
            if other_label.casefold() == label.casefold():
                continue
            other_position = segment.casefold().find(other_label.casefold())
            if other_position >= 0:
                boundaries.append(other_position)
        total_position = segment.casefold().find("total")
        if total_position >= 0:
            boundaries.append(total_position)
        if boundaries:
            segment = segment[: min(boundaries)]
        segment = segment[:160]
        if _CURRENCY.search(segment):
            return segment
    return None


def extract_allocations(text: str, labels: Iterable[str], minimum_value: float = 100_000) -> dict[str, float]:
    labels = list(labels)
    allocations: dict[str, float] = {}
    for label in labels:
        segment = _allocation_segment(text, label, labels)
        if segment is None:
            continue
        values = [parse_currency(match.group(0)) for match in _CURRENCY.finditer(segment)]
        major_values = [value for value in values if value >= minimum_value]
        if major_values:
            lowered = segment.casefold()
            if "current" in lowered and ("recommended" in lowered or "proposed" in lowered) and len(major_values) > 1:
                allocations[label] = major_values[1]
            else:
                allocations[label] = major_values[0]
    return allocations


def validate_word_limit(text: str, limit: int) -> ValidationFinding:
    count = word_count(text)
    return ValidationFinding(
        validator="word_limit",
        passed=count <= limit,
        severity="error",
        message=f"Word count {count} is {'within' if count <= limit else 'over'} the limit of {limit}.",
        details={"word_count": count, "limit": limit},
    )


def validate_allocation_total(text: str, spec: dict[str, Any]) -> ValidationFinding:
    labels = spec["labels"]
    expected = float(spec["expected_total"])
    tolerance = float(spec.get("tolerance", 1.0))
    allocations = extract_allocations(text, labels, float(spec.get("minimum_value", 100_000)))
    missing = [label for label in labels if label not in allocations]
    total = sum(allocations.values())
    passed = not missing and abs(total - expected) <= tolerance
    if missing:
        message = f"Could not resolve allocations for: {', '.join(missing)}."
    else:
        message = f"Resolved allocation total is ${total:,.0f}; expected ${expected:,.0f}."
    return ValidationFinding(
        validator="allocation_total",
        passed=passed,
        severity="error",
        message=message,
        details={"allocations": allocations, "total": total, "expected_total": expected, "missing": missing},
    )


def validate_risk_patterns(text: str, patterns: list[dict[str, str]]) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for index, spec in enumerate(patterns, start=1):
        match = re.search(spec["pattern"], text, flags=re.IGNORECASE | re.DOTALL)
        findings.append(
            ValidationFinding(
                validator=f"risk_pattern_{index}",
                passed=match is None,
                severity=spec.get("severity", "warning"),
                message=spec["message"] if match else f"Risk pattern {index} not detected.",
                details={"matched_text": match.group(0)[:240] if match else None},
            )
        )
    return findings


def validate_candidate(task: Task, text: str) -> list[ValidationFinding]:
    constraints = task.constraints
    findings: list[ValidationFinding] = []
    if "max_words" in constraints:
        findings.append(validate_word_limit(text, int(constraints["max_words"])))
    if "allocation_total" in constraints:
        findings.append(validate_allocation_total(text, constraints["allocation_total"]))
    if constraints.get("risk_patterns"):
        findings.extend(validate_risk_patterns(text, constraints["risk_patterns"]))
    return findings

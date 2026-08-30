from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RubricItem:
    id: str
    description: str
    max_points: float


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    prompt: str
    rubric: tuple[RubricItem, ...]
    constraints: dict[str, Any] = field(default_factory=dict)

    @property
    def max_quality_score(self) -> float:
        return sum(item.max_points for item in self.rubric)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Task":
        rubric = tuple(RubricItem(**item) for item in value.get("rubric", []))
        return cls(
            id=value["id"],
            title=value["title"],
            prompt=value["prompt"],
            rubric=rubric,
            constraints=value.get("constraints", {}),
        )


@dataclass
class Generation:
    provider: str
    model: str
    text: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    id: str
    text: str
    generation: Generation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        result = {"id": self.id, "text": self.text, "metadata": self.metadata}
        if self.generation:
            result["generation"] = self.generation.to_mapping()
        return result


@dataclass(frozen=True)
class ValidationFinding:
    validator: str
    passed: bool
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeResult:
    judge: str
    candidate_id: str
    scores: dict[str, float]
    fatal_errors: list[str]
    summary: str
    latency_seconds: float | None = None
    estimated_cost_usd: float | None = None

    @property
    def total(self) -> float:
        return sum(self.scores.values())

    def to_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["total"] = self.total
        return result

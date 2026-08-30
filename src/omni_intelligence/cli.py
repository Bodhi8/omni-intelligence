from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .orchestrator import OmniIntelligence
from .providers import provider_from_mapping
from .types import Task
from .validators import validate_candidate


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | None, value: Any) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False)
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _pilot(task_path: str, candidates_path: str, output: str | None) -> int:
    task = Task.from_mapping(_read_json(task_path))
    records = _read_json(candidates_path)
    ranking = []
    for record in records:
        scores = {key: float(value) for key, value in record.get("human_scores", {}).items()}
        subtotal = sum(scores.values())
        penalties = sum(float(value) for value in record.get("penalties", {}).values())
        ranking.append(
            {
                "candidate_id": record["id"],
                "model_alias": record.get("model_alias", "anonymous"),
                "latency_seconds": record.get("latency_seconds"),
                "human_scores": scores,
                "subtotal": subtotal,
                "penalties": record.get("penalties", {}),
                "total": subtotal + penalties,
                "validation": [finding.to_mapping() for finding in validate_candidate(task, record["text"])],
            }
        )
    ranking.sort(key=lambda row: row["total"], reverse=True)
    result = {
        "system": "OMNI — Orchestrated Multi-Model Networked Inference",
        "mode": "replay_pilot",
        "task_id": task.id,
        "method": "Human-defined rubric with deterministic constraint checks",
        "limitations": [
            "Four responses are an illustrative pilot, not a statistically powered benchmark.",
            "Only one response included latency, so latency comparisons are incomplete.",
            "Model identities remain blinded in the supplied data.",
        ],
        "ranking": ranking,
    }
    _write_json(output, result)
    for index, row in enumerate(ranking, start=1):
        latency = "unknown" if row["latency_seconds"] is None else f'{row["latency_seconds"]:.1f}s'
        print(f'{index}. {row["candidate_id"]}: {row["total"]:.1f}/100 ({latency})')
    return 0


async def _run_live(task_path: str, models_path: str, output: str | None) -> int:
    task = Task.from_mapping(_read_json(task_path))
    configuration = _read_json(models_path)
    solvers = [provider_from_mapping(spec) for spec in configuration.get("solvers", []) if spec.get("enabled", True)]
    judges = [provider_from_mapping(spec) for spec in configuration.get("judges", []) if spec.get("enabled", True)]
    synth_spec = configuration.get("synthesizer")
    synthesizer = provider_from_mapping(synth_spec) if synth_spec and synth_spec.get("enabled", True) else None
    if not solvers:
        raise SystemExit("At least one enabled solver is required")
    result = await OmniIntelligence(
        seed=int(configuration.get("seed", 41)),
        quality_weight=float(configuration.get("quality_weight", 0.9)),
        max_rounds=int(configuration.get("max_rounds", 3)),
        quality_threshold=float(configuration.get("quality_threshold", 90.0)),
        convergence_threshold=float(configuration.get("convergence_threshold", 0.55)),
        judge_disagreement_threshold=float(configuration.get("judge_disagreement_threshold", 0.12)),
        min_improvement=float(configuration.get("min_improvement", 1.0)),
        plateau_patience=int(configuration.get("plateau_patience", 1)),
        max_estimated_cost_usd=configuration.get("max_estimated_cost_usd"),
    ).run(
        task=task,
        solvers=solvers,
        judges=judges,
        synthesizer=synthesizer,
        top_k=int(configuration.get("top_k", 3)),
    )
    _write_json(output, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omni-intelligence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pilot = subparsers.add_parser("pilot", help="Replay and validate a saved comparison")
    pilot.add_argument("--task", required=True)
    pilot.add_argument("--candidates", required=True)
    pilot.add_argument("--output")

    live = subparsers.add_parser("run", help="Run a live multi-model decision mesh")
    live.add_argument("--task", required=True)
    live.add_argument("--models", required=True)
    live.add_argument("--output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "pilot":
        return _pilot(args.task, args.candidates, args.output)
    return asyncio.run(_run_live(args.task, args.models, args.output))


if __name__ == "__main__":
    raise SystemExit(main())

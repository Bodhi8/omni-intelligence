from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from typing import Any

from .providers import ModelProvider
from .scoring import (
    adjusted_quality,
    confidence_from_quality_and_disagreement,
    disagreement_index,
    efficiency_score,
    extract_json_object,
    hard_constraints_pass,
    lexical_convergence,
    normalize_judge_result,
)
from .types import Candidate, JudgeResult, Task, ValidationFinding
from .validators import validate_candidate


SOLVER_SYSTEM = """You are an independent decision analyst. Solve the task using only the supplied evidence.
Do not invent missing facts. Give a concise decision rationale, not private chain-of-thought. Follow every output constraint."""

JUDGE_SYSTEM = """You are a blinded evaluator. Evaluate the candidate only against the task, evidence, rubric, and explicit constraints.
Do not infer the candidate's model identity. Penalize invented facts, invalid calculations, and false precision. Return JSON only."""

ARBITER_SYSTEM = """You are the downstream arbiter in a recursive multi-model decision system.
Identify supported common ground, unresolved disagreements, invalid calculations, missing constraints, and the highest-value revisions.
Agreement is not proof. Do not preserve a claim merely because several models repeat it. Give concise, actionable feedback."""

SYNTHESIS_SYSTEM = """You are the final synthesis layer in a recursive multi-model decision system. Produce one final answer using the strongest supported elements.
Correct errors identified by validators and critics. Agreement is not proof. Do not repeat a claim unless the supplied evidence supports it."""


class OmniIntelligence:
    def __init__(
        self,
        seed: int = 41,
        quality_weight: float = 0.9,
        max_rounds: int = 3,
        quality_threshold: float = 90.0,
        convergence_threshold: float = 0.55,
        judge_disagreement_threshold: float = 0.12,
        min_improvement: float = 1.0,
        plateau_patience: int = 1,
        max_estimated_cost_usd: float | None = None,
    ):
        self._rng = random.Random(seed)
        self.quality_weight = quality_weight
        self.max_rounds = max(1, max_rounds)
        self.quality_threshold = quality_threshold
        self.convergence_threshold = convergence_threshold
        self.judge_disagreement_threshold = judge_disagreement_threshold
        self.min_improvement = min_improvement
        self.plateau_patience = max(1, plateau_patience)
        self.max_estimated_cost_usd = max_estimated_cost_usd

    def _revision_prompt(
        self,
        task: Task,
        prior_candidates: list[Candidate],
        feedback: str,
        solver_index: int,
    ) -> str:
        shared = [
            {"candidate_id": candidate.id, "response": candidate.text}
            for candidate in prior_candidates
        ]
        prior_match = next(
            (candidate for candidate in prior_candidates if candidate.metadata.get("solver_index") == solver_index),
            None,
        )
        prior = prior_match.text if prior_match else "No prior response was returned by this solver."
        return f"""ORIGINAL TASK
{task.prompt}

YOUR PRIOR RESPONSE
{prior}

ANONYMIZED FRONTIER OUTPUTS
{json.dumps(shared, indent=2)}

DOWNSTREAM ARBITER FEEDBACK
{feedback}

Produce a revised answer independently. Correct supported errors, retain defensible minority insights, and obey the original constraints.
Do not converge merely for consensus and do not mention this optimization loop."""

    async def _generate_candidates(
        self,
        task: Task,
        solvers: list[ModelProvider],
        round_index: int = 0,
        prior_candidates: list[Candidate] | None = None,
        feedback: str = "",
    ) -> list[Candidate]:
        prompts = [
            task.prompt
            if prior_candidates is None
            else self._revision_prompt(task, prior_candidates, feedback, index)
            for index in range(len(solvers))
        ]
        generations = await asyncio.gather(
            *(solver.generate(prompt, SOLVER_SYSTEM) for solver, prompt in zip(solvers, prompts, strict=True)),
            return_exceptions=True,
        )
        candidates: list[Candidate] = []
        for index, generation in enumerate(generations, start=1):
            if isinstance(generation, Exception):
                continue
            candidates.append(
                Candidate(
                    id=f"r{round_index:02d}-candidate-{index:03d}",
                    text=generation.text,
                    generation=generation,
                    metadata={"round": round_index, "solver_index": index - 1},
                )
            )
        if not candidates:
            raise RuntimeError("No solver returned a usable candidate")
        return candidates

    def _judge_prompt(self, task: Task, candidate: Candidate) -> str:
        rubric = [
            {"id": item.id, "description": item.description, "max_points": item.max_points}
            for item in task.rubric
        ]
        return f"""TASK
{task.prompt}

RUBRIC
{json.dumps(rubric, indent=2)}

CANDIDATE ID
{candidate.id}

CANDIDATE RESPONSE
{candidate.text}

Return exactly this JSON structure:
{{
  "scores": {{"rubric_item_id": 0}},
  "fatal_errors": ["unsupported or disqualifying error"],
  "summary": "brief evidence-based assessment"
}}
"""

    async def _judge_candidates(
        self,
        task: Task,
        candidates: list[Candidate],
        judges: list[ModelProvider],
    ) -> list[JudgeResult]:
        if not judges:
            return []
        work: list[tuple[ModelProvider, Candidate]] = []
        shuffled = candidates[:]
        self._rng.shuffle(shuffled)
        for judge in judges:
            judge_order = shuffled[:]
            self._rng.shuffle(judge_order)
            work.extend((judge, candidate) for candidate in judge_order)
        generations = await asyncio.gather(
            *(judge.generate(self._judge_prompt(task, candidate), JUDGE_SYSTEM) for judge, candidate in work),
            return_exceptions=True,
        )
        results: list[JudgeResult] = []
        for (judge, candidate), generation in zip(work, generations, strict=True):
            if isinstance(generation, Exception):
                continue
            try:
                payload = extract_json_object(generation.text)
                results.append(
                    normalize_judge_result(
                        judge_name=judge.name,
                        candidate_id=candidate.id,
                        payload=payload,
                        rubric=task.rubric,
                        latency_seconds=generation.latency_seconds,
                        estimated_cost_usd=generation.estimated_cost_usd,
                    )
                )
            except (ValueError, TypeError):
                continue
        return results

    def _rank(
        self,
        task: Task,
        candidates: list[Candidate],
        judgments: list[JudgeResult],
    ) -> list[dict[str, Any]]:
        by_candidate: dict[str, list[JudgeResult]] = defaultdict(list)
        for judgment in judgments:
            by_candidate[judgment.candidate_id].append(judgment)
        ranking: list[dict[str, Any]] = []
        for candidate in candidates:
            findings = validate_candidate(task, candidate.text)
            candidate_judgments = by_candidate[candidate.id]
            quality = adjusted_quality(candidate_judgments, findings) if candidate_judgments else 0.0
            disagreement = disagreement_index(candidate_judgments, task.max_quality_score or 100.0)
            latency = candidate.generation.latency_seconds if candidate.generation else None
            cost = candidate.generation.estimated_cost_usd if candidate.generation else None
            efficiency = efficiency_score(latency, cost)
            utility = self.quality_weight * quality + (1.0 - self.quality_weight) * efficiency
            ranking.append(
                {
                    "candidate_id": candidate.id,
                    "quality": quality,
                    "efficiency": efficiency,
                    "utility": utility,
                    "disagreement": disagreement,
                    "confidence": confidence_from_quality_and_disagreement(quality, disagreement),
                    "validation": [finding.to_mapping() for finding in findings],
                    "judgments": [judgment.to_mapping() for judgment in candidate_judgments],
                    "generation": candidate.generation.to_mapping() if candidate.generation else None,
                }
            )
        return sorted(ranking, key=lambda row: (row["utility"], row["quality"]), reverse=True)

    def _feedback_prompt(
        self,
        task: Task,
        candidates: list[Candidate],
        ranking: list[dict[str, Any]],
    ) -> str:
        by_id = {candidate.id: candidate for candidate in candidates}
        packets = []
        for row in ranking:
            packets.append(
                {
                    "candidate_id": row["candidate_id"],
                    "response": by_id[row["candidate_id"]].text,
                    "quality_score": row["quality"],
                    "validator_failures": [
                        finding["message"] for finding in row["validation"] if not finding["passed"]
                    ],
                    "critic_summaries": [judgment["summary"] for judgment in row["judgments"]],
                    "fatal_errors": [
                        error for judgment in row["judgments"] for error in judgment["fatal_errors"]
                    ],
                }
            )
        return f"""ORIGINAL TASK
{task.prompt}

CURRENT FRONTIER OUTPUTS AND AUDITS
{json.dumps(packets, indent=2)}

Return a feedback memo with four short sections: VERIFIED COMMON GROUND, UNSUPPORTED CLAIMS, UNRESOLVED DISAGREEMENTS, and NEXT-ROUND DIRECTIVES.
Do not produce the final user answer in this step."""

    def _round_state(
        self,
        round_index: int,
        candidates: list[Candidate],
        ranking: list[dict[str, Any]],
        prior_best_quality: float | None,
        cumulative_cost: float,
    ) -> dict[str, Any]:
        best = ranking[0]
        convergence = lexical_convergence([candidate.text for candidate in candidates])
        mean_judge_disagreement = (
            sum(row["disagreement"] for row in ranking) / len(ranking) if ranking else 0.0
        )
        improvement = None if prior_best_quality is None else best["quality"] - prior_best_quality
        constraints_pass = hard_constraints_pass(
            [ValidationFinding(**finding) for finding in best["validation"]]
        )
        verified_convergence = (
            constraints_pass
            and best["quality"] >= self.quality_threshold
            and convergence >= self.convergence_threshold
            and mean_judge_disagreement <= self.judge_disagreement_threshold
        )
        return {
            "round": round_index,
            "ranking": ranking,
            "best_quality": best["quality"],
            "quality_improvement": improvement,
            "frontier_convergence": convergence,
            "mean_judge_disagreement": mean_judge_disagreement,
            "hard_constraints_pass": constraints_pass,
            "verified_convergence": verified_convergence,
            "cumulative_estimated_cost_usd": cumulative_cost,
        }

    def _synthesis_prompt(
        self,
        task: Task,
        candidates: list[Candidate],
        ranking: list[dict[str, Any]],
        top_k: int,
    ) -> str:
        by_id = {candidate.id: candidate for candidate in candidates}
        packets: list[dict[str, Any]] = []
        for row in ranking[:top_k]:
            candidate = by_id[row["candidate_id"]]
            packets.append(
                {
                    "candidate_id": candidate.id,
                    "response": candidate.text,
                    "validator_failures": [
                        finding["message"] for finding in row["validation"] if not finding["passed"]
                    ],
                    "critic_summaries": [judgment["summary"] for judgment in row["judgments"]],
                    "fatal_errors": [
                        error for judgment in row["judgments"] for error in judgment["fatal_errors"]
                    ],
                }
            )
        return f"""ORIGINAL TASK
{task.prompt}

ANONYMIZED CANDIDATES AND AUDITS
{json.dumps(packets, indent=2)}

Write the final answer. Preserve only supported claims, satisfy every task constraint, and do not mention this orchestration process."""

    async def run(
        self,
        task: Task,
        solvers: list[ModelProvider],
        judges: list[ModelProvider],
        synthesizer: ModelProvider | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        rounds: list[dict[str, Any]] = []
        candidates: list[Candidate] | None = None
        feedback = ""
        prior_best_quality: float | None = None
        plateau_count = 0
        cumulative_cost = 0.0
        stop_reason = "maximum_rounds"

        for round_index in range(self.max_rounds):
            candidates = await self._generate_candidates(
                task=task,
                solvers=solvers,
                round_index=round_index,
                prior_candidates=candidates,
                feedback=feedback,
            )
            cumulative_cost += sum(
                candidate.generation.estimated_cost_usd or 0.0
                for candidate in candidates
                if candidate.generation
            )
            judgments = await self._judge_candidates(task, candidates, judges)
            cumulative_cost += sum(judgment.estimated_cost_usd or 0.0 for judgment in judgments)
            ranking = self._rank(task, candidates, judgments)
            state = self._round_state(
                round_index,
                candidates,
                ranking,
                prior_best_quality,
                cumulative_cost,
            )
            rounds.append(state)

            if state["verified_convergence"]:
                stop_reason = "verified_convergence"
                break
            improvement = state["quality_improvement"]
            if improvement is not None and improvement < self.min_improvement:
                plateau_count += 1
            else:
                plateau_count = 0
            if plateau_count >= self.plateau_patience and state["hard_constraints_pass"]:
                stop_reason = "quality_plateau"
                break
            if self.max_estimated_cost_usd is not None and cumulative_cost >= self.max_estimated_cost_usd:
                stop_reason = "cost_ceiling"
                break
            if round_index + 1 >= self.max_rounds:
                break
            prior_best_quality = state["best_quality"]
            if synthesizer:
                arbiter_generation = await synthesizer.generate(
                    self._feedback_prompt(task, candidates, ranking),
                    ARBITER_SYSTEM,
                )
                feedback = arbiter_generation.text
                cumulative_cost += arbiter_generation.estimated_cost_usd or 0.0
            else:
                feedback = "Use the validator and critic feedback attached to the anonymized candidates."

        if candidates is None:
            raise RuntimeError("The recursive loop produced no candidates")
        ranking = rounds[-1]["ranking"]
        final_candidate: Candidate | None = None
        if synthesizer:
            generation = await synthesizer.generate(
                self._synthesis_prompt(task, candidates, ranking, min(top_k, len(candidates))),
                SYNTHESIS_SYSTEM,
            )
            final_candidate = Candidate(id="synthesis-001", text=generation.text, generation=generation)
            cumulative_cost += generation.estimated_cost_usd or 0.0
        return {
            "system": "OMNI — Orchestrated Multi-Model Networked Inference",
            "task_id": task.id,
            "architecture": "recursive_frontier_mesh",
            "candidate_count_per_round": len(candidates),
            "judge_count": len(judges),
            "round_count": len(rounds),
            "stop_reason": stop_reason,
            "cumulative_estimated_cost_usd": cumulative_cost,
            "rounds": rounds,
            "ranking": ranking,
            "final": {
                "candidate": final_candidate.to_mapping(),
                "validation": [finding.to_mapping() for finding in validate_candidate(task, final_candidate.text)],
            }
            if final_candidate
            else None,
        }

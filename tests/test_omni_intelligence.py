import json
import unittest
from pathlib import Path

from omni_intelligence.orchestrator import OmniIntelligence
from omni_intelligence.providers import ReplayProvider
from omni_intelligence.scoring import lexical_convergence
from omni_intelligence.types import Task
from omni_intelligence.validators import extract_allocations, validate_candidate


ROOT = Path(__file__).resolve().parents[1]


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = Task.from_mapping(
            json.loads((ROOT / "examples" / "marketing_budget_task.json").read_text(encoding="utf-8"))
        )
        cls.candidates = json.loads(
            (ROOT / "examples" / "pilot_candidates.json").read_text(encoding="utf-8")
        )

    def test_all_pilot_allocations_resolve_to_one_million(self):
        labels = ["Paid Search", "Paid Social", "Email/CRM"]
        for candidate in self.candidates:
            with self.subTest(candidate=candidate["id"]):
                allocations = extract_allocations(candidate["text"], labels)
                self.assertEqual(set(allocations), set(labels))
                self.assertEqual(sum(allocations.values()), 1_000_000)

    def test_known_unsupported_search_calculation_is_flagged(self):
        findings = validate_candidate(self.task, self.candidates[2]["text"])
        risk = next(finding for finding in findings if finding.validator == "risk_pattern_1")
        self.assertFalse(risk.passed)

    def test_disciplined_response_avoids_known_risk_patterns(self):
        findings = validate_candidate(self.task, self.candidates[0]["text"])
        risk_findings = [finding for finding in findings if finding.validator.startswith("risk_pattern")]
        self.assertTrue(all(finding.passed for finding in risk_findings))


class ConvergenceTests(unittest.TestCase):
    def test_identical_outputs_fully_converge(self):
        self.assertEqual(lexical_convergence(["A supported answer", "A supported answer"]), 1.0)

    def test_unrelated_outputs_do_not_converge(self):
        score = lexical_convergence(["causal budget allocation", "protein folding laboratory"])
        self.assertEqual(score, 0.0)


class RecursiveLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_verified_convergence_stops_the_loop(self):
        task = Task.from_mapping(
            json.loads((ROOT / "examples" / "marketing_budget_task.json").read_text(encoding="utf-8"))
        )
        candidate_text = json.loads(
            (ROOT / "examples" / "pilot_candidates.json").read_text(encoding="utf-8")
        )[0]["text"]
        judge_text = json.dumps(
            {
                "scores": {
                    "accuracy": 20,
                    "reasoning_quality": 24,
                    "decision_quality": 14,
                    "uncertainty_handling": 10,
                    "experiment_design": 9,
                    "instruction_following": 10,
                    "efficiency": 9,
                },
                "fatal_errors": [],
                "summary": "Supported and constraint-compliant.",
            }
        )
        solvers = [
            ReplayProvider("solver-a", "replay", candidate_text),
            ReplayProvider("solver-b", "replay", candidate_text),
        ]
        judges = [
            ReplayProvider("judge-a", "replay", judge_text),
            ReplayProvider("judge-b", "replay", judge_text),
        ]
        synthesizer = ReplayProvider("arbiter", "replay", candidate_text)
        result = await OmniIntelligence(max_rounds=3).run(task, solvers, judges, synthesizer)
        self.assertEqual(result["stop_reason"], "verified_convergence")
        self.assertEqual(result["round_count"], 1)
        self.assertTrue(all(item["passed"] for item in result["final"]["validation"] if item["severity"] == "error"))


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest

from app.agents.synthesis_agent import agent_synthesis_evaluator


def review_state(candidate_nav_ic50: float) -> dict:
    return {
        "original_smiles": "CCO",
        "modified_smiles": "CCCO",
        "toxicity_prob": 0.8,
        "is_toxic": True,
        "risk_level": "high",
        "channel_predictions": {
            "herg": {"ic50_uM": 1.0, "pIC50": 6.0},
            "cav1.2": {"ic50_uM": 20.0, "pIC50": 4.7},
            "nav1.5": {"ic50_uM": 10.0, "pIC50": 5.0},
        },
        "modified_toxicity_prob": 0.1,
        "modified_is_toxic": False,
        "modified_risk_level": "low",
        "modified_toxicity_source": "test",
        "modified_channel_predictions": {
            "herg": {"ic50_uM": 2.0, "pIC50": 5.7},
            "cav1.2": {"ic50_uM": 1.0, "pIC50": 6.0},
            "nav1.5": {"ic50_uM": candidate_nav_ic50, "pIC50": 5.0},
        },
        "modified_tdp": {"risk": "low"},
        "optimization_round": 1,
        "max_optimization_rounds": 3,
        "optimization_history": [],
        "candidate_history": [],
        "agent_trace": [],
        "rag_context": [],
        "modified_rag_context": [],
    }


class MultiObjectiveReviewTests(unittest.TestCase):
    def test_candidate_passes_all_constraints(self):
        result = asyncio.run(agent_synthesis_evaluator(review_state(10.0)))
        self.assertTrue(result["optimization_success"])

    def test_nav15_ic50_decrease_rejects_candidate(self):
        result = asyncio.run(agent_synthesis_evaluator(review_state(5.0)))
        self.assertFalse(result["optimization_success"])
        self.assertIn("Nav1.5 IC50", result["failure_reason"])


if __name__ == "__main__":
    unittest.main()

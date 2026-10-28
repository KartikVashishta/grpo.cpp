import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/countdown"))
from report import pair_answers


class PairedAnswersTest(unittest.TestCase):
    def setUp(self):
        self.metrics = [{"run": "base-42", "step": 0, "implementation": "native", "seed": 42},
                        {"run": "trained-42", "step": 64, "implementation": "native", "seed": 42}]
        self.runs = [{"name": "trained-42", "container": {"function_seconds": 143, "estimated_compute_usd": .053}}]
        self.answers = []
        for i, (before, after) in enumerate(((0, 1), (1, 0), (0, 0), (1, 1))):
            for metric, reward in zip(self.metrics, (before, after)):
                self.answers.append({"run": metric["run"], "step": metric["step"], "id": str(i),
                                     "nums": [1, 2, 3], "target": i, "reward": reward,
                                     "text": "1 + 2 + 3", "value": "6", "status": "correct" if reward else "wrong_value"})

    def test_pairs_different_run_names_by_puzzle(self):
        pairs = pair_answers(self.metrics, list(reversed(self.answers)), self.runs)
        outcomes = {p["id"]: p["outcome"] for p in pairs[0]["puzzles"]}
        self.assertEqual(outcomes, {"0": "improved", "1": "regressed", "2": "still_wrong", "3": "still_correct"})
        self.assertEqual(pairs[0]["name"], "trained-42")

    def test_rejects_missing_or_duplicate_answers(self):
        for answers in (self.answers[:-1], self.answers + [self.answers[0]]):
            with self.subTest(length=len(answers)), self.assertRaises(ValueError):
                pair_answers(self.metrics, answers, self.runs)

    def test_rejects_changed_puzzle(self):
        answers = copy.deepcopy(self.answers)
        answers[1]["target"] = 99
        with self.assertRaises(ValueError):
            pair_answers(self.metrics, answers, self.runs)


if __name__ == "__main__":
    unittest.main()

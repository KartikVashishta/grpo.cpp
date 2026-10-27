import itertools
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/countdown"))
from task import make_splits, puzzle_id, verify
from pilot import summarize


class CountdownTest(unittest.TestCase):
    def test_exact_arithmetic_and_number_usage(self):
        cases = [
            ("(8 + 7) * 3", [3, 7, 8], 45, "correct"),
            ("8 / (3 - 1)", [8, 3, 1], 4, "correct"),
            ("1 / 3 * 3", [1, 3, 3], 1, "correct"),
            ("8 + 7 + 3", [3, 7, 8], 45, "wrong_value"),
            ("45", [3, 7, 8], 45, "wrong_numbers"),
            ("3 + 3 + 8", [3, 7, 8], 14, "wrong_numbers"),
            ("8 / (3 - 3)", [8, 3, 3], 1, "division_by_zero"),
        ]
        for text, numbers, target, expected in cases:
            with self.subTest(text=text):
                result = verify(text, numbers, target)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.reward, float(expected == "correct"))

    def test_forbidden_syntax_never_earns_reward(self):
        for text in ["", "The answer is (8+7)*3", "(8+7)*3=45", "```(8+7)*3```",
                     "3**7+8", "8//3+7", "3.0+7+8", "-3+7+8", "3_0+7+8",
                     "True+7+8", "__import__('os').system('true')", "(3,7,8)",
                     "3+7+8 # answer", "(" * 300 + "3" + ")" * 300]:
            with self.subTest(text=text[:40]):
                self.assertEqual(verify(text, [3, 7, 8], 45).status, "parse_error")

    def test_permutations_and_source_order_cannot_leak(self):
        rows = [{"nums": list(p), "target": 45} for p in itertools.permutations([3, 7, 8])]
        rows += [{"nums": [i, i + 1, i + 2], "target": 3 * i + 3} for i in range(1, 12)]
        sizes = {"calibration": 2, "dev": 2, "test": 2, "train": 6}
        splits, count = make_splits(rows, sizes, 42)
        self.assertEqual(count, 12)
        self.assertEqual(make_splits(list(reversed(rows)), sizes, 42)[0], splits)
        ids = [p["id"] for values in splits.values() for p in values]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids.count(puzzle_id([8, 3, 7], 45)), 1)
        with self.assertRaises(ValueError):
            make_splits(rows, {"train": 13}, 42)

    def test_learning_signal_is_measured_per_prompt(self):
        records = []
        for key, rewards in [("unsolved", [0] * 8), ("solved", [1] * 8),
                             ("mixed", [1] + [0] * 7)]:
            records.extend({"id": key, "reward": reward, "tokens": 10,
                            "eos": True, "status": "correct" if reward else "wrong_value"}
                           for reward in rewards)
        stats = summarize(records)
        self.assertEqual(stats["mixed_groups"], 1)
        self.assertEqual(stats["prompts"], 3)
        self.assertEqual(stats["accuracy"], 9 / 24)
        self.assertEqual(stats["generated_tokens"], 240)

    @unittest.skipUnless(importlib.util.find_spec("modal"), "requires the Modal Python environment")
    def test_runner_imports_at_the_remote_path(self):
        for name in ("modal_run", "native_run"):
            path = Path(__file__).resolve().parents[1] / f"experiments/countdown/{name}.py"
            remote = f"/root/{name}.py"
            with self.subTest(name=name), patch("modal.is_local", return_value=False):
                exec(compile(path.read_text(), remote, "exec"),
                     {"__name__": name, "__file__": remote})


if __name__ == "__main__":
    unittest.main()

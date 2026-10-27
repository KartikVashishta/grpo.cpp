"""Countdown's prompt, exact verifier and permutation-safe split rule."""

import ast
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import operator
import re


@dataclass(frozen=True)
class Verdict:
    status: str
    value: str | None = None

    @property
    def reward(self):
        return float(self.status == "correct")

    def as_dict(self):
        return asdict(self)


def verify(text, numbers, target):
    """Accept only integer literals, parentheses and binary +, -, *, /.

    Fractions keep division exact. No expression is executed as Python code.
    Unary signs, decimals, extra prose and partially extracted answers fail.
    """
    text = text.strip()
    if not text or len(text) > 512 or not re.fullmatch(r"[0-9+*/()\s-]+", text):
        return Verdict("parse_error")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, RecursionError):
        return Verdict("parse_error")
    used = []
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            used.append(node.value)
            if len(used) > len(numbers):
                raise ValueError
            return Fraction(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError

    try:
        value = evaluate(tree.body)
    except ZeroDivisionError:
        return Verdict("division_by_zero")
    except (ValueError, RecursionError):
        return Verdict("parse_error")
    if Counter(used) != Counter(numbers):
        return Verdict("wrong_numbers", str(value))
    return Verdict("correct" if value == target else "wrong_value", str(value))


def puzzle_id(numbers, target):
    return ",".join(map(str, sorted(numbers))) + ":" + str(target)


def make_splits(rows, sizes, seed):
    """Deduplicate puzzle identities before selecting any split."""
    puzzles = {}
    for row in rows:
        numbers, target = row["nums"], row["target"]
        if len(numbers) != 3:
            continue
        if type(target) is not int or any(type(n) is not int or n <= 0 for n in numbers):
            raise ValueError("Countdown requires positive integer inputs and an integer target")
        key = puzzle_id(numbers, target)
        puzzles[key] = {"id": key, "nums": sorted(numbers), "target": target}
    ranked = sorted(puzzles, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    if any(type(n) is not int or n <= 0 for n in sizes.values()):
        raise ValueError("Split sizes must be positive integers")
    if sum(sizes.values()) > len(ranked):
        raise ValueError("Not enough distinct three-number puzzles")
    result, offset = {}, 0
    for name, count in sizes.items():
        result[name] = [puzzles[key] for key in ranked[offset:offset + count]]
        offset += count
    return result, len(ranked)


def messages(puzzle):
    return [
        {"role": "system", "content":
         "Solve the arithmetic puzzle. Use every given number exactly once. "
         "Use only +, -, *, / and parentheses. Output one expression and nothing else."},
        {"role": "user", "content": "Numbers: 3, 7, 8\nTarget: 45"},
        {"role": "assistant", "content": "(8 + 7) * 3"},
        {"role": "user", "content": "Numbers: 2, 4, 16\nTarget: 2"},
        {"role": "assistant", "content": "16 / 4 - 2"},
        {"role": "user", "content":
         f"Numbers: {', '.join(map(str, puzzle['nums']))}\nTarget: {puzzle['target']}"},
    ]


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()

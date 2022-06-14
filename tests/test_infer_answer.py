import string
from typing import List

import numpy as np
import rich
from parameterized import parameterized

from medical_reasoning.models.functional.infer_answer import infer_answer_from_choices


@parameterized([
    ("A", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("B", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "B"),
    ("A) ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("A, ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("A: ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("A. ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("A ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("B) ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "B"),
    ("Belgium A) ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("Africa B) ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "B"),
    ("Belgium A: ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("Africa B: ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "B"),
    ("Belgium A. ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("Africa B. ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "B"),
    ("A) ..., B) ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("B) ..., B) ...", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "B"),
    ("_option_A_", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("_option_B_", "reasoning", ["A", "B"], ["_option_A_", "_option_B_"], "B"),
    ("...", "... A ", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("...", "... B ", ["A", "B"], ["_option_A_", "_option_B_"], "B"),
    ("...", "...B... A ", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("...", "...A... B ", ["A", "B"], ["_option_A_", "_option_B_"], "B"),
    ("...", "...B... _option_A_", ["A", "B"], ["_option_A_", "_option_B_"], "A"),
    ("...", "...A... _option_B_", ["A", "B"], ["_option_A_", "_option_B_"], "B"),
    ("...", "...", ["A", "B"], ["_option_A_", "_option_B_"], None),
])
def test_infer_answer_from_choices(
    answer: str,
    pre_answer: str,
    options_symbols: List[str],
    options: List[str],
    expected: int
):
    output = infer_answer_from_choices(
        answer, pre_answer=pre_answer, option_symbols=options_symbols, options=options)
    if expected is None:
        assert output is None
    else:
        assert output == expected

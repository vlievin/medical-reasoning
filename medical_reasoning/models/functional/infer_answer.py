from __future__ import annotations

import re
from typing import Iterable
from typing import List
from typing import Optional
from typing import T

import rich
from loguru import logger


def get_start_indices(target: str | List, pattern: str) -> list[int]:
    matches = re.finditer(pattern, target)
    return [m.start() for m in matches]


def safe_min(lst: Iterable[T]) -> Optional[T]:
    if len(lst):
        return min(lst)
    else:
        return None


def get_first_match(query, *, choices, keys, op=min):
    assert len(choices) == len(keys)
    indices = [(key, get_start_indices(query, o)) for key, o in zip(keys, choices)]
    indices = list(filter(lambda x: len(x[1]), indices))
    if len(indices):
        return op(indices, key=lambda x: x[1])[0]
    else:
        return None


def infer_answer_from_choices(
    prompt_answer: str,
    *,
    option_symbols: Optional[List] = None,
    options: Optional[List] = None,
    pre_answer: Optional[str] = None,
) -> None | str:
    # step 1. Try to cache the options from `self.options`
    match = get_first_match(
        prompt_answer, choices=option_symbols, keys=option_symbols, op=min
    )
    if match is not None:
        return match

    # step 2. Try to cache the options from `options`
    logger.debug(
        f"Inferring labels from {option_symbols} failed. "
        f"trying to match the provided options"
    )
    match = get_first_match(prompt_answer, choices=options, keys=option_symbols, op=min)
    if match is not None:
        return match
    elif pre_answer is None:
        return None

    # step 3. Try to catch a last mention of the answer in the pre-answer
    logger.debug(
        f"Inferring labels from {options} failed. " f"trying to match the pre answer"
    )
    match = get_first_match(pre_answer, choices=options, keys=option_symbols, op=max)
    if match is not None:
        return match

    match = get_first_match(
        pre_answer, choices=option_symbols, keys=option_symbols, op=max
    )
    if match is not None:
        return match

    logger.warning(f"Failed to match any answer ({prompt_answer})")
    rich.print(f">> prompt_answer: {prompt_answer}")
    rich.print(f">> pre_answer: {pre_answer}")
    rich.print(f">> options: {options}")

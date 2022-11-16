from __future__ import annotations

import re
from typing import Iterable
from typing import List
from typing import Optional
from typing import T

import rich
from loguru import logger


def get_start_indices(target: str | List, pattern: str, warn:bool=True,) -> list[int]:
    try:
        matches = re.finditer(pattern, target)
        return [m.start() for m in matches]
    except Exception as exc:
        if warn:
            logger.warning(f"Failed to infer answer: {exc}")
        return []


def safe_min(lst: Iterable[T]) -> Optional[T]:
    if len(lst):
        return min(lst)
    else:
        return None


def get_first_match(query, *, choices, keys, op=min, warn:bool=True,):
    assert len(choices) == len(keys)
    indices = [(key, get_start_indices(query, o, warn=warn)) for key, o in zip(keys, choices)]
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
    warn:bool=True
) -> None | str:

    if len(prompt_answer) == 1:
        return prompt_answer

    # make the regex patterns for the option symbols
    option_symbols_re = [rf"{o}(\)|:|\.|,| )" for o in option_symbols]

    # step 1. Try to cache the options from `self.options`
    match = get_first_match(
        prompt_answer, choices=option_symbols_re, keys=option_symbols, op=min, warn=warn,
    )
    if match is not None:
        return match

    # step 2. Try to cache the options from `options`
    logger.debug(
        f"Inferring labels from {option_symbols} failed. "
        f"trying to match the provided options"
    )
    match = get_first_match(prompt_answer, choices=options, keys=option_symbols, op=min,  warn=warn,)
    if match is not None:
        return match
    elif pre_answer is None:
        return None

    # step 3. Try to catch a last mention of the answer in the pre-answer
    logger.debug(
        f"Inferring labels from {options} failed. " f"trying to match the pre answer"
    )
    match = get_first_match(pre_answer, choices=options, keys=option_symbols, op=max,  warn=warn,)
    if match is not None:
        return match

    match = get_first_match(
        pre_answer, choices=option_symbols_re, keys=option_symbols, op=max,  warn=warn,
    )
    if match is not None:
        return match

    if warn:
        logger.warning(f"Failed to match any answer ({prompt_answer})")
    if "<GPT-3-answer>" not in prompt_answer:
        logger.debug(f"prompt_answer: {prompt_answer}")
        logger.debug(f"pre_answer: {pre_answer}")
        logger.debug(f"options: {options}")

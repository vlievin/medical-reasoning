from __future__ import annotations

import abc
import re
from abc import ABC
from typing import Any
from typing import List
from typing import Optional
from typing import T

import rich
from loguru import logger

LINE_BRAKE = "\n"


def get_start_indices(target: str | List, pattern: str) -> list[int]:
    matches = re.finditer(pattern, target)
    return [m.start() for m in matches]


def safe_min(lst: T) -> Optional[T]:
    if len(lst):
        return min(lst)
    else:
        return None


class PromptTemplate(object):
    @abc.abstractmethod
    def format_question(self, *args, **kwargs) -> str:
        raise NotImplementedError()


class ChainOfThoughtTemplate(PromptTemplate, ABC):
    zero_shot_prompt = ...
    reasoning_prompt = ...
    extractive_prompt = ...

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"zero_shot_prompt={self.zero_shot_prompt}, "
            f"reasoning_prompt={self.reasoning_prompt}, "
            f"extractive_prompt={self.extractive_prompt}"
            f")"
        )

    def make_zero_shot_prompt(self, *args, **kwargs) -> str:
        question_prompt = self.format_question(*args, **kwargs)
        return f"{question_prompt}\n\n{self.zero_shot_prompt}"

    def make_reasoning_prompt(self, *args, **kwargs) -> str:
        question_prompt = self.format_question(*args, **kwargs)
        return f"{question_prompt}\n\n{self.reasoning_prompt}"

    def make_extractive_prompt(self, completed_prompt: str) -> str:
        return f"{completed_prompt}\n\n{self.extractive_prompt}"

    def infer_answer(
        self,
        extractive_prompt_answer,
        *,
        options: Optional[List] = None,
        pre_answer: Optional[str] = None,
    ) -> Any:
        ...


class MultipleChoiceTemplate(ChainOfThoughtTemplate):
    def __init__(
        self,
        options=None,
        identity: str = "medical expert",
    ):
        if options is None:
            options = ["A", "B", "C", "D", "E"]
        self.options = options
        if identity in ("none", "null"):
            identity = None
        if identity is not None:
            identity = identity.replace("_", " ").strip()
        self.identity = identity

    @property
    def reasoning_prompt(self):
        prompt = "Answer: Let's think step by step"
        if self.identity is not None:
            prompt = f"{prompt} like a {self.identity}"
        return f"{prompt}."

    @property
    def options_reasoning_prompt(self):
        min_opt = self.options[0]
        max_opt = self.options[-1]
        prompt = f"Now, let's reflect on each option ({min_opt} through {max_opt})."
        # prompt = f"Now, let's reflect on each option (from the least likely
        # to the most likely option)."
        # prompt = f"Now, let's eliminate options one-by-one until
        # we find the correct one."
        return prompt

    def make_option_reasoning_prompt(self, completed_prompt: str) -> str:
        return f"{completed_prompt}\n\n{self.options_reasoning_prompt}"

    @property
    def extractive_prompt(self):
        min_opt = self.options[0]
        max_opt = self.options[-1]
        # return f"Therefore, all information considered,
        # among {min_opt} through {max_opt}, the answer is"
        return f"Therefore, among {min_opt} through {max_opt}, the answer is"

    @property
    def zero_shot_prompt(self):
        min_opt = self.options[0]
        max_opt = self.options[-1]
        return f"A: among {min_opt} through {max_opt}, the answer is"

    def format_question(self, question: str, options: List[str]) -> str:
        formatted_options = [
            f"{self.options[i]}) {option}" for i, option in enumerate(options)
        ]
        return f"Question: {question}\n\nAnswer options:\n{LINE_BRAKE.join(formatted_options)}"

    def infer_answer(
        self,
        prompt_answer: str,
        *,
        options: Optional[List] = None,
        pre_answer: Optional[str] = None,
    ) -> None | str:

        # step 1. Try to cache the options from `self.options`
        indices = [(o, get_start_indices(prompt_answer, o)) for o in self.options]
        indices = list(filter(lambda x: len(x[1]), indices))
        if len(indices):
            return min(indices, key=lambda x: x[1])[0]
        elif options is None:
            return None

        # step 2. Try to cache the options from `options`
        logger.debug(
            f"> Inferring  labels from {self.options} failed. "
            f"trying to match the provided options"
        )

        indices = [
            (o, get_start_indices(prompt_answer, o_))
            for o, o_ in zip(self.options, options)
        ]
        indices = list(filter(lambda x: len(x[1]), indices))
        if len(indices):
            return min(indices, key=lambda x: x[1])[0]
        elif pre_answer is None:
            return None

        # step 3. Try to catch a last mention of the answer in the pre-answer
        logger.debug(
            f"> Inferring  labels from {options} failed. "
            f"trying to match the pre answer"
        )
        indices = [
            (o, get_start_indices(pre_answer, o_))
            for o, o_ in zip(self.options, options)
        ]
        indices = list(filter(lambda x: len(x[1]), indices))
        if len(indices):
            return max(indices, key=lambda x: x[1])[0]

        indices = [(o, get_start_indices(prompt_answer, o)) for o in self.options]
        indices = list(filter(lambda x: len(x[1]), indices))
        if len(indices):
            return max(indices, key=lambda x: x[1])[0]

        logger.warning(f"Failed to match any answer ({prompt_answer})")
        rich.print(f">> prompt_answer: {prompt_answer}")
        rich.print(f">> pre_answer: {pre_answer}")
        rich.print(f">> options: {options}")

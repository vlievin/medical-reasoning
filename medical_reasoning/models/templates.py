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

from medical_reasoning.models.functional.infer_answer import infer_answer_from_choices

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
        use_documents: bool = False,
    ):
        if options is None:
            options = ["A", "B", "C", "D", "E"]
        self.options = options
        if identity in ("none", "null"):
            identity = None
        if identity is not None:
            identity = identity.replace("_", " ").strip()
        self.identity = identity
        self.use_documents = use_documents

    @property
    def reasoning_prompt(self):
        min_opt = self.options[0]
        max_opt = self.options[-1]
        prompt = "Answer: Let's think step by step"
        if self.identity is not None:
            prompt = f"{prompt} like a {self.identity}"
        f"{prompt}  to arrive at one of the options {min_opt} through {max_opt}"  # todo
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

    def format_question(
        self,
        question: str,
        options: List[str],
        documents: Optional[List] = None,
        **kwargs,
    ) -> str:
        prompt = ""

        if self.use_documents:
            if documents is None:
                raise ValueError("documents must be provided if use_documents is True")
            formatted_documents = "\n".join(documents)
            prompt += f"Context: {formatted_documents}\n\n"

        formatted_options = [
            f"{self.options[i]}) {option}" for i, option in enumerate(options)
        ]
        prompt += f"Question: {question}\n\nAnswer options:\n{LINE_BRAKE.join(formatted_options)}"

        return prompt

    def infer_answer(
        self,
        prompt_answer: str,
        *,
        options: Optional[List] = None,
        pre_answer: Optional[str] = None,
    ) -> None | str:

        return infer_answer_from_choices(
            prompt_answer,
            options=options,
            option_symbols=self.options,
            pre_answer=pre_answer,
        )

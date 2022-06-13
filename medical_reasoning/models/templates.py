from __future__ import annotations

import abc
import re
from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import T

from medical_reasoning.models.functional.infer_answer import infer_answer_from_choices
from medical_reasoning.utils.datastruct import Example

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
    name = "prompt"
    SEP = "\n\n"
    can_be_simulated = True
    _completion_config = {}

    @abc.abstractmethod
    def __call__(self, eg: Example) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def simulate_completion(self, eg: Example, **kargs) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def infer_answer(
        self,
        prompt_answer: str,
        *,
        eg: Example,
        pre_answer: Optional[str] = None,
        **kwargs,
    ) -> Any:
        raise NotImplementedError()

    @property
    def completion_config(self) -> Dict:
        return copy(self._completion_config)

    def __repr__(self):
        return type(self).__name__

    @property
    def description(self) -> str:
        return self.__class__.__name__


class MultipleChoiceTemplate(PromptTemplate):
    name = "multiple_choice_prompt"
    first_symbol_pattern = "::A::"
    last_symbol_pattern = "::D::"

    def __init__(
        self,
        use_documents: bool = False,
    ):
        self.use_documents = use_documents

    def __call__(self, eg: Example) -> str:
        steps = [self.format_question(eg), self.zero_shot_prompt(eg)]
        steps = [s for s in steps if len(s)]
        return self.SEP.join(steps)

    @staticmethod
    def zero_shot_prompt(eg: Example):
        return (
            f"Answer: among {eg.allowed_options[0]} "
            f"through {eg.allowed_options[-1]}, the answer is"
        )

    def format_question(self, eg: Example) -> str:
        prompt = ""

        if self.use_documents:
            if eg.documents is None or len(eg.documents) == 0:
                raise ValueError("documents must be provided if use_documents is True")
            formatted_documents = "\n".join(eg.documents)
            prompt += f"Context: {formatted_documents}\n\n"

        formatted_options = [
            f"{eg.allowed_options[i]}) {option}" for i, option in enumerate(eg.options)
        ]
        prompt += (
            f"Question: {eg.question}{self.SEP}{LINE_BRAKE.join(formatted_options)}"
        )

        return prompt

    def infer_answer(
        self,
        prompt_answer: str,
        *,
        eg: Example,
        pre_answer: Optional[str] = None,
        **kwargs,
    ) -> None | str:

        return infer_answer_from_choices(
            prompt_answer,
            options=eg.options,
            option_symbols=eg.allowed_options,
            pre_answer=pre_answer,
        )

    def simulate_completion(self, eg: Example) -> str:
        return f" {eg.answer_symbol}) {eg.answer}."


class ReasoningMultipleChoiceTemplate(MultipleChoiceTemplate):
    name = "reasoning_prompt"

    def __init__(self, strategy: str = "Let's think step by step", **kwargs):
        super(ReasoningMultipleChoiceTemplate, self).__init__(**kwargs)
        if strategy in ("none", "null"):
            strategy = None
        if strategy is not None:
            strategy = strategy.replace("_", " ").strip()
        self.strategy = strategy

    @property
    def description(self) -> str:
        return f"{self.strategy}"

    def format_strategy(self, eg: Example) -> str:
        strategy = copy(self.strategy)
        strategy = strategy.replace(self.first_symbol_pattern, eg.allowed_options[0])
        strategy = strategy.replace(self.last_symbol_pattern, eg.allowed_options[-1])
        return strategy

    def __call__(self, eg: Example) -> str:
        steps = [self.format_question(eg), self.format_strategy(eg)]
        steps = [s for s in steps if len(s)]
        return self.SEP.join(steps)

    def simulate_completion(self, eg: Example) -> str:
        return f"\n{eg.reasoning}"

    def __repr__(self):
        return f'{type(self).__name__}("{self.strategy}")'

    def infer_answer(
        self,
        prompt_answer: str,
        *,
        eg: Example,
        pre_answer: Optional[str] = None,
        **kwargs,
    ) -> None | str:
        return None


class ExtractionMultipleChoiceTemplate(MultipleChoiceTemplate):
    name = "extractive_prompt"
    _completion_config = {"max_tokens": 32, "n": 1}

    def __call__(self, eg: Example) -> str:
        steps = [self.extractive_prompt(eg)]
        steps = [s for s in steps if len(s)]
        return self.SEP.join(steps)

    @staticmethod
    def extractive_prompt(eg: Example) -> str:
        return (
            f"\n\nTherefore, among {eg.allowed_options[0]} "
            f"through {eg.allowed_options[-1]}, the answer is"
        )

    def simulate_completion(self, eg: Example) -> str:
        return f" {eg.answer_symbol}) {eg.answer}."


class UncertaintyTemplate(PromptTemplate):
    name = "uncertainty_prompt"
    can_be_simulated = False
    _completion_config = {"max_tokens": 32, "n": 1}

    def __call__(self, eg: Example) -> str:
        steps = [self.uncertainty_prompt(eg)]
        steps = [s for s in steps if len(s)]
        return self.SEP.join(steps)

    @staticmethod
    def uncertainty_prompt(eg: Example) -> str:
        return "\nConfidence (from 1 to 5) and second most likely answer: "

    def simulate_completion(self, eg: Example) -> str:
        raise NotImplementedError(
            "simulate_completion cannot be called on UncertaintyTemplate"
        )

    def infer_answer(
        self,
        prompt_answer: str,
        *,
        eg: Example,
        pre_answer: Optional[str] = None,
        **kwargs,
    ) -> dict:

        # extract the answer
        answer = infer_answer_from_choices(
            prompt_answer,
            options=eg.options,
            option_symbols=eg.allowed_options,
            pre_answer=None,
        )

        # extract the confidence
        pseudo_eg = Example(
            question="",
            options=5 * [""],
            documents=[],
            reasoning=None,
            allowed_options=["1", "2", "3", "4", "5"],
            answer_idx=-1,
        )
        confidence = infer_answer_from_choices(
            prompt_answer,
            options=pseudo_eg.options,
            option_symbols=pseudo_eg.allowed_options,
            pre_answer=None,
        )

        return {"confidence": confidence, "second_answer": answer}

from __future__ import annotations

import abc
import hashlib
import re
from collections import OrderedDict
from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import T

import numpy as np

from medical_reasoning.models.functional.infer_answer import infer_answer_from_choices
from medical_reasoning.utils.datastruct import Example

ACCEPTED_STYLES = {"full2", "full", "short", "mmlu", "none"}


def format_option(symbol, option):
    return f"{symbol}) {option}"


def format_option_2(symbol, option):
    return f"({symbol}) {option}"


def get_start_indices(target: str | List, pattern: str) -> list[int]:
    matches = re.finditer(pattern, target)
    return [m.start() for m in matches]


def safe_min(lst: T) -> Optional[T]:
    if len(lst):
        return min(lst)
    else:
        return None


class AnswerChoicesFormat:
    @staticmethod
    def style_1(eg: Example):
        return f"among {eg.option_symbols[0]} through {eg.option_symbols[-1]}"

    def style_2(eg: Example):
        return (
            f"between {', '.join(eg.option_symbols[0:-1])} or {eg.option_symbols[-1]}"
        )

    def style_3(eg: Example):
        x = hashlib.sha256(str(eg.uid).encode("utf-8"))
        seed = int(x.hexdigest(), base=16) % 2 ** 32
        rgn = np.random.RandomState(seed)
        option_symbols = copy(eg.option_symbols)
        rgn.shuffle(option_symbols)
        return f"between {', '.join(option_symbols[0:-1])} or {option_symbols[-1]}"


class PromptTemplate(object):
    name = "prompt"
    _completion_config = {}

    def __init__(self, *, style: str = "full"):
        self._style = style
        style_parts = self._style.split("-")
        if len(style_parts) > 1:
            style, answer_style = style_parts
        else:
            style, answer_style = style_parts[0], "1"

        self.answer_fmt = {
            "1": AnswerChoicesFormat.style_1,
            "2": AnswerChoicesFormat.style_2,
            "3": AnswerChoicesFormat.style_3,
        }[answer_style]

        if style not in ACCEPTED_STYLES:
            raise ValueError(
                f"style {style} is not recognized. Accepted styles are: {ACCEPTED_STYLES}"
            )
        self.style = style
        self.separator = {
            "full2": "\n\n",
            "full": "\n\n",
            "short": "\n\n",
            "mmlu": "\n",
            "none": "\n\n",
        }[self.style]

    @abc.abstractmethod
    def __call__(self, eg: Example) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def simulate_completion(self, eg: Example, **kargs) -> str:
        raise NotImplementedError()

    def can_be_simulated(self, eg: Example) -> bool:
        return False

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_documents = use_documents

    def __call__(self, eg: Example) -> str:
        steps = [self.format_question(eg), self.zero_shot_prompt(eg)]
        steps = [s for s in steps if len(s)]
        return self.separator.join(steps)

    @property
    def description(self) -> str:
        return "--"

    def zero_shot_prompt(self, eg: Example):

        # select the preprompt
        prepromt = {
            "full2": "Answer: ",
            "full": "Answer: ",
            "short": "A: ",
            "mmlu": "A: ",
            "none": "",
        }[self.style]

        return f"{prepromt}{self.answer_fmt(eg)}, the answer is"

    def format_question(self, eg: Example) -> str:
        prompt = ""

        if self.use_documents:
            if eg.documents is None or len(eg.documents) == 0:
                raise ValueError("documents must be provided if use_documents is True")
            docs = eg.documents
            if len(docs) == len(eg.option_symbols):
                if len(set(docs)) == 1:
                    # if all documents are identical, use the first one
                    docs = [docs[0]]
                else:
                    # if there is one document per answer option, assume
                    # each document was sampled
                    # for each option, and add a `Document <option>` at
                    # the beginning of each document
                    docs = [
                        f"Document {o}. {doc}"
                        for doc, o in zip(eg.documents, eg.option_symbols)
                    ]

            formatted_documents = "\n".join(docs)
            prompt += f"Context: {formatted_documents}{self.separator}"

        opt_format = {
            "full2": format_option_2,
            "full": format_option,
            "short": format_option,
            "mmlu": format_option_2,
            "none": format_option,
        }[self.style]

        formatted_options = [
            opt_format(eg.option_symbols[i], option)
            for i, option in enumerate(eg.options)
        ]

        # select the preprompt
        question_prepromt = {
            "full2": "Question: ",
            "full": "Question: ",
            "short": "Q: ",
            "mmlu": "Q: ",
            "none": "",
        }[self.style]

        option_prepromt = {
            "full2": f"Answer choices:\n",
            "full": self.separator,
            "short": self.separator,
            "mmlu": "\n",
            "none": self.separator,
        }[self.style]

        option_sep = {
            "full2": "\n",
            "full": "\n",
            "short": "\n",
            "mmlu": " ",
            "none": "\n",
        }[self.style]

        prompt += (
            f"{question_prepromt}{eg.question}{option_prepromt}"
            f"{option_sep.join(formatted_options)}"
        )

        return prompt

    def infer_answer(
        self,
        prompt_answer: str,
        *,
        eg: Example,
        pre_answer: Optional[str] = None,
        warn: bool = True,
        **kwargs,
    ) -> None | str:

        pred = infer_answer_from_choices(
            prompt_answer,
            options=eg.options,
            option_symbols=eg.option_symbols,
            pre_answer=pre_answer,
            warn=warn,
        )
        return pred

    def simulate_completion(self, eg: Example) -> str:
        return f" {eg.answer_symbol}) {eg.answer}."

    def can_be_simulated(self, eg: Example) -> bool:
        return True


class ReasoningMultipleChoiceTemplate(MultipleChoiceTemplate):
    name = "reasoning_prompt"

    def __init__(self, strategy: str = "Let's think step by step", **kwargs):
        super(ReasoningMultipleChoiceTemplate, self).__init__(**kwargs)
        if strategy in ("none", "null", "--"):
            strategy = None
        if strategy is not None:
            strategy = strategy.replace("_", " ").strip()
        self.strategy = strategy

    @property
    def description(self) -> str:
        return f"{self.strategy}"

    def format_strategy(self, eg: Example) -> str:
        # format the strategy
        strategy = copy(self.strategy)
        strategy = strategy.replace(self.first_symbol_pattern, eg.option_symbols[0])
        strategy = strategy.replace(self.last_symbol_pattern, eg.option_symbols[-1])

        # select the preprompt
        prepromt = {
            "full2": "Answer: ",
            "full": "Answer: ",
            "short": "A: ",
            "mmlu": "A: ",
            "none": "",
        }[self.style]

        return f"{prepromt}{strategy}"

    def __call__(self, eg: Example) -> str:
        steps = [self.format_question(eg), self.format_strategy(eg)]
        steps = [s for s in steps if len(s)]
        return self.separator.join(steps)

    def infer_answer(
            self,
            prompt_answer: str,
            *,
            warn: bool = False,
            **kwargs,
    ) -> None | str:
        COT_ANSWER_PATTERN = "the answer is"
        if COT_ANSWER_PATTERN in prompt_answer:
            prompt_answer = prompt_answer.split(COT_ANSWER_PATTERN)[1]
        return super().infer_answer(prompt_answer, warn=warn, **kwargs)

    def simulate_completion(self, eg: Example) -> str:
        return f"\n{eg.reasoning}"

    def can_be_simulated(self, eg: Example) -> bool:
        return eg.reasoning is not None and len(eg.reasoning) > 0

    def __repr__(self):
        return f'{type(self).__name__}("{self.strategy}")'


class ExtractionMultipleChoiceTemplate(MultipleChoiceTemplate):
    name = "extractive_prompt"
    _completion_config = {"max_tokens": 32, "n": 1}

    def __call__(self, eg: Example) -> str:
        steps = [self.extractive_prompt(eg)]
        steps = [s for s in steps if len(s)]
        return self.separator.join(steps)

    def extractive_prompt(self, eg: Example) -> str:
        return f"\n\nTherefore, {self.answer_fmt(eg)}, the answer is"

    def simulate_completion(self, eg: Example) -> str:
        return f" {eg.answer_symbol}) {eg.answer}."

    @property
    def description(self) -> str:
        return "extraction"


class UncertaintyTemplate(PromptTemplate):
    name = "uncertainty_prompt"
    _completion_config = {"max_tokens": 32, "n": 1}

    def __call__(self, eg: Example) -> str:
        steps = [self.uncertainty_prompt(eg)]
        steps = [s for s in steps if len(s)]
        return self.separator.join(steps)

    def uncertainty_prompt(self, eg: Example) -> str:
        prepromt = {
            "full2": "Confidence: ",
            "full": "Confidence: ",
            "short": "",
            "mmlu": "",
            "none": "",
        }[self.style]
        return (
            f"\n\n{prepromt}Rate your confidence on a grade from 1 to 5 "
            "and give the second most likely answer:"
        )

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
            option_symbols=eg.option_symbols,
            pre_answer=None,
        )

        # extract the confidence
        pseudo_eg = Example(
            question="",
            uid="",
            options=5 * [""],
            documents=[],
            reasoning=None,
            option_symbols=["1", "2", "3", "4", "5"],
            answer_idx=-1,
        )
        confidence = infer_answer_from_choices(
            prompt_answer,
            options=pseudo_eg.options,
            option_symbols=pseudo_eg.option_symbols,
            pre_answer=None,
        )

        return {"confidence": confidence, "second_answer": answer}

    @property
    def description(self) -> str:
        return "uncertainty"


def auto_templates(**templates) -> OrderedDict:
    """Handle special cases when building chains of templates"""
    templates = [(name, templates) for name, templates in templates.items()]
    first_template = templates[0][1]
    if (
        isinstance(first_template, ReasoningMultipleChoiceTemplate)
        and first_template.strategy is None
    ):
        assert isinstance(templates[1][1], ExtractionMultipleChoiceTemplate)
        direct_template = MultipleChoiceTemplate(
            use_documents=first_template.use_documents, style=first_template._style
        )
        templates = [("direct", direct_template)] + templates[2:]

    return OrderedDict(templates)

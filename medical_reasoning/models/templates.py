import abc
import string
from abc import ABC
from typing import Any
from typing import List

LINE_BRAKE = "\n"


def safe_index(lst: List, value: Any, default_index: Any) -> Any:
    if value in lst:
        return lst.index(value)
    else:
        return default_index


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
        return f"{completed_prompt} {self.extractive_prompt}"

    @staticmethod
    def infer_answer(self, extractive_prompt_answer) -> Any:
        ...


class MultipleChoiceTemplate(ChainOfThoughtTemplate):
    zero_shot_prompt = "A: among A through D, the answer is "
    reasoning_prompt = "A: Let's think step by step like a medical expert."
    extractive_prompt = "Therefore, among A through D, the answer is "

    def __init__(self, options=None):
        if options is None:
            options = ["A", "B", "C", "D", "E"]
        self.options = options

    @staticmethod
    def format_question(question: str, options: List[str]) -> str:
        formatted_options = [
            f"{string.ascii_uppercase[i]}) {option}" for i, option in enumerate(options)
        ]
        return f"Q: {question}\n\n{LINE_BRAKE.join(formatted_options)}"

    def infer_answer(self, extractive_prompt_answer: str) -> str:
        indices = [
            (o, safe_index(extractive_prompt_answer, o, None)) for o in self.options
        ]
        indices = list(filter(lambda x: x[1] is not None, indices))
        if len(indices):
            return min(indices, key=lambda x: x[1])[0]
        else:
            return None

import abc
import os
import string
from abc import ABC
from typing import Any
from typing import Dict
from typing import List

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

LINE_BRAKE = "\n"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


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
            options = ["A", "B", "C", "D"]
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


class Reasoner(object):
    def __init__(
        self,
        *,
        engine: str = "text-ada-001",
        prompt_mode="chain_of_thought",
        template: ChainOfThoughtTemplate = ...,
    ):
        self.engine = engine
        self.template = template
        self.prompt_mode = prompt_mode
        assert self.prompt_mode in {"chain_of_thought", "zero_shot"}

    def __call__(self, *args, **kwargs) -> (str, Dict[str, Any]):
        diagnostics = {}
        if self.prompt_mode == "chain_of_thought":
            # reasoning step
            reasoning_prompt = self.template.make_reasoning_prompt(*args, **kwargs)
            reasoning_answer = self._get_prompt_completion(reasoning_prompt)
            completed_prompt = reasoning_prompt + reasoning_answer
            diagnostics["reasoning"] = reasoning_answer.strip()

            # extractive step
            extractive_prompt = self.template.make_extractive_prompt(completed_prompt)
            extractive_answer = self._get_prompt_completion(extractive_prompt)
            completed_prompt = extractive_prompt + extractive_answer
            diagnostics["answer"] = extractive_answer.strip()

            # extract the answer and return
            answer = self.template.infer_answer(extractive_answer)
            diagnostics["completed_prompt"] = completed_prompt
            return answer, diagnostics
        elif self.prompt_mode == "zero_shot":
            zero_shot_prompt = self.template.make_zero_shot_prompt(*args, **kwargs)
            zero_shot_answer = self._get_prompt_completion(zero_shot_prompt)

            # extract the answer and return
            answer = self.template.infer_answer(zero_shot_answer)
            full_answer = zero_shot_prompt + zero_shot_answer
            diagnostics["completed_prompt"] = full_answer
            return answer, diagnostics
        else:
            raise ValueError(f"Unknown prompt mode: {self.prompt_mode}")

    def _get_prompt_completion(self, prompt) -> str:
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=0,
            max_tokens=300,
            top_p=1,
            logprobs=5,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["<|endoftext|>"],
        )
        completion = response["choices"][0]["text"]
        return completion.split("<|endoftext|>")[0]

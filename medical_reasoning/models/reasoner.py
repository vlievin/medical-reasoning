import os
from typing import Any
from typing import Dict
from typing import List

import openai
import rich
from dotenv import load_dotenv

from medical_reasoning.models.templates import ChainOfThoughtTemplate
from medical_reasoning.models.templates import MultipleChoiceTemplate

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


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
        assert self.prompt_mode in {
            "option_chain_of_thought",
            "chain_of_thought",
            "zero_shot",
        }

    def __repr__(self):
        return (
            f"Reasoner("
            f"engine={self.engine}, "
            f"prompt_mode={self.prompt_mode}, "
            f"template={self.template})"
        )

    def __call__(
        self, *args, options: List[str] = None, **kwargs
    ) -> (str, Dict[str, Any]):
        diagnostics = {}
        if self.prompt_mode == "chain_of_thought":
            # reasoning step
            reasoning_prompt = self.template.make_reasoning_prompt(
                *args, options=options, **kwargs
            )
            reasoning_answer = self._get_prompt_completion(reasoning_prompt)
            completed_prompt = reasoning_prompt + reasoning_answer
            diagnostics["reasoning"] = reasoning_answer.strip()

            # extractive step
            extractive_prompt = self.template.make_extractive_prompt(completed_prompt)
            extractive_answer = self._get_prompt_completion(
                extractive_prompt,
                stop=["<|endoftext|>"],
                max_tokens=32,
            )
            completed_prompt = extractive_prompt + extractive_answer
            diagnostics["answer"] = extractive_answer.strip()

            # extract the answer and return
            answer = self.template.infer_answer(
                extractive_answer, options=options, pre_answer=reasoning_answer
            )
            diagnostics["completed_prompt"] = completed_prompt
            return answer, diagnostics
        elif self.prompt_mode == "option_chain_of_thought":
            if not isinstance(self.template, MultipleChoiceTemplate):
                raise TypeError(
                    f"{self.prompt_mode} is only "
                    f"compatible with MultipleChoiceTemplate"
                )

            # reasoning step
            reasoning_prompt = self.template.make_reasoning_prompt(
                *args, options=options, **kwargs
            )
            reasoning_answer = self._get_prompt_completion(reasoning_prompt)
            completed_prompt = reasoning_prompt + reasoning_answer
            diagnostics["reasoning"] = reasoning_answer.strip()

            # option evaluation step
            option_eval_prompt = self.template.make_option_reasoning_prompt(
                completed_prompt
            )
            option_eval_answer = self._get_prompt_completion(option_eval_prompt)
            completed_prompt = option_eval_prompt + option_eval_answer
            diagnostics["option_eval"] = option_eval_answer.strip()

            # extractive step
            extractive_prompt = self.template.make_extractive_prompt(completed_prompt)
            extractive_answer = self._get_prompt_completion(
                extractive_prompt,
                max_tokens=32,
            )
            completed_prompt = extractive_prompt + extractive_answer
            diagnostics["answer"] = extractive_answer.strip()

            # extract the answer and return
            answer = self.template.infer_answer(
                extractive_answer, options=options, pre_answer=reasoning_answer
            )
            diagnostics["completed_prompt"] = completed_prompt
            return answer, diagnostics
        elif self.prompt_mode == "zero_shot":
            zero_shot_prompt = self.template.make_zero_shot_prompt(
                *args, options=options, **kwargs
            )
            zero_shot_answer = self._get_prompt_completion(zero_shot_prompt)
            diagnostics["answer"] = zero_shot_answer.strip()

            # extract the answer and return
            answer = self.template.infer_answer(zero_shot_answer, options=options)
            full_answer = zero_shot_prompt + zero_shot_answer
            diagnostics["completed_prompt"] = full_answer
            return answer, diagnostics
        else:
            raise ValueError(f"Unknown prompt mode: {self.prompt_mode}")

    def _get_prompt_completion(
        self, prompt, stop="<|endoftext|>", max_tokens=512
    ) -> str:
        if self.engine == "dryrun":
            return "\n<GPT-3-answer>\n"
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=0,  # todo: increase
            max_tokens=max_tokens,
            # max_tokens=512,
            top_p=1,
            # logprobs=5,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )
        completion = response["choices"][0]["text"]
        return completion.split("<|endoftext|>")[0]

import os
from typing import Any
from typing import Dict
from typing import List
import os
import openai
import rich
from dotenv import load_dotenv
from transformers import GPT2Tokenizer

from medical_reasoning.models.templates import ChainOfThoughtTemplate
from medical_reasoning.models.templates import MultipleChoiceTemplate
from medical_reasoning.utils.datastruct import Example

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Reasoner(object):
    def __init__(
        self,
        *,
        engine: str = "text-ada-001",
        prompt_mode="chain_of_thought",
        template: ChainOfThoughtTemplate = ...,
        price:float,
        tokenizer: GPT2Tokenizer,
    ):
        self.engine = engine
        self.template = template
        self.price = price
        self.tokenizer = tokenizer
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
        self, eg: Example, shots: List[Example], **kwargs
    ) -> (str, Dict[str, Any]):
        completed_prompt = ""
        for shot in shots:
            _, meta = self.process_example(
                shot, completed_prompt=completed_prompt, simulate=True, **kwargs
            )
            completed_prompt = meta["completed_prompt"]
            completed_prompt += "\n\n"

        return self.process_example(eg, completed_prompt=completed_prompt, **kwargs)

    def process_example(
        self, eg: Example, simulate: bool = False, completed_prompt: str = ""
    ) -> (str, Dict[str, Any]):
        diagnostics = {}
        if self.prompt_mode == "chain_of_thought":
            # reasoning step
            reasoning_prompt = self.template.make_reasoning_prompt(
                eg.question,
                options=eg.options,
                documents=eg.documents,
            )
            reasoning_prompt = completed_prompt + reasoning_prompt
            if simulate:
                gold_reasoning = eg.reasoning
                if gold_reasoning is None or len(gold_reasoning) == 0:
                    raise ValueError("Reasoning must be known to run simulations")
                reasoning_answer = f"\n{gold_reasoning}"
            else:
                reasoning_answer = self._get_prompt_completion(reasoning_prompt)
            # todo: make this cleaner
            if not simulate or len(reasoning_answer) > 10:
                completed_prompt = reasoning_prompt + reasoning_answer
            diagnostics["reasoning"] = reasoning_answer.strip()

            # extractive step
            extractive_prompt = self.template.make_extractive_prompt(completed_prompt)
            if simulate:
                extractive_answer = f" {eg.answer_symbol}) {eg.answer}."
            else:
                extractive_answer = self._get_prompt_completion(
                    extractive_prompt,
                    max_tokens=32,
                )
            completed_prompt = extractive_prompt + extractive_answer
            diagnostics["answer"] = extractive_answer.strip()

            # extract the answer and return
            answer = self.template.infer_answer(
                extractive_answer, options=eg.options, pre_answer=reasoning_answer
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
                eg.question,
                options=eg.options,
                documents=eg.documents,
            )
            reasoning_prompt = completed_prompt + reasoning_prompt
            if simulate:
                gold_reasoning = eg.reasoning
                if gold_reasoning is None or len(gold_reasoning) == 0:
                    raise ValueError("Reasoning must be known to run simulations")
                reasoning_answer = f"\n{gold_reasoning}"
            else:
                reasoning_answer = self._get_prompt_completion(reasoning_prompt)
            # todo: make this cleaner
            if not simulate or len(reasoning_answer) > 10:
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
            if simulate:
                extractive_answer = f" {eg.answer_symbol}) {eg.answer}."
            else:
                extractive_answer = self._get_prompt_completion(
                    extractive_prompt,
                    max_tokens=32,
                )
            completed_prompt = extractive_prompt + extractive_answer
            diagnostics["answer"] = extractive_answer.strip()

            # extract the answer and return
            answer = self.template.infer_answer(
                extractive_answer, options=eg.options, pre_answer=reasoning_answer
            )
            diagnostics["completed_prompt"] = completed_prompt
            return answer, diagnostics
        elif self.prompt_mode == "zero_shot":
            zero_shot_prompt = self.template.make_zero_shot_prompt(
                eg.question,
                options=eg.options,
                documents=eg.documents,
            )
            zero_shot_prompt = completed_prompt + zero_shot_prompt
            if simulate:
                zero_shot_answer = f" {eg.answer_symbol}) {eg.answer}."
            else:
                zero_shot_answer = self._get_prompt_completion(
                    zero_shot_prompt,
                    max_tokens=32,
                )

            diagnostics["answer"] = zero_shot_answer.strip()

            # extract the answer and return
            answer = self.template.infer_answer(zero_shot_answer, options=eg.options)
            full_answer = zero_shot_prompt + zero_shot_answer
            diagnostics["completed_prompt"] = full_answer
            return answer, diagnostics
        else:
            raise ValueError(f"Unknown prompt mode: {self.prompt_mode}")

    def _get_prompt_completion(
        self, prompt, stop="<|endoftext|>", max_tokens=512
    ) -> str:
        if self.engine == "dryrun":
            return "<GPT-3-answer>"
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

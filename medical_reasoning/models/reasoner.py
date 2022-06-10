import os
import time
from copy import copy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import openai
import rich
import yaml
from dotenv import load_dotenv
from transformers import GPT2Tokenizer

from medical_reasoning import configs
from medical_reasoning.models.cache import CachedFunction
from medical_reasoning.models.templates import MultipleChoiceTemplate
from medical_reasoning.models.templates import PromptTemplate
from medical_reasoning.utils.datastruct import Example

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_CONFIG_PATH = (
    Path(configs.__file__).parent / "model" / "config" / "default.yaml"
)


class Reasoner(object):
    def __init__(
        self,
        *,
        engine: str = "text-ada-001",
        templates: Dict[str, PromptTemplate],
        price: float,
        tokenizer: GPT2Tokenizer,
        max_rate: float = 60,
        cache_dir: os.PathLike = None,
        config: Optional[Dict] = None,
    ):

        self.engine = engine
        self.templates = templates
        self.price = price
        self.tokenizer = tokenizer
        self.max_rate = max_rate
        self.cache = CachedFunction(cache_dir=cache_dir)

        if config is None:
            config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, "r").read())
        self.config = config
        rich.print(f"> Config:\n{self.config}")

        # state
        self.reset_stats()

    def get_engine_args(self, **kwargs) -> Dict:
        cfg = copy(self.config)
        cfg.update(kwargs)
        return cfg

    def reset_stats(self):
        self.total_cost = 0
        self.n_calls = 0
        self.calls_timestamps = []

    def timestamp(self):
        self.n_calls += 1
        self.calls_timestamps.append(time.time())
        self.calls_timestamps = self.last_calls(period=3600)

    def last_calls(self, period=60) -> List:
        now = time.time()
        time_stamps = [ts for ts in self.calls_timestamps if now - ts < period]
        return time_stamps

    @property
    def is_dryrun(self) -> bool:
        return bool(os.getenv("DRYRUN", default=False))

    def __repr__(self):
        return f"Reasoner(" f"engine={self.engine}, " f"templates={self.templates})"

    def __call__(
        self, eg: Example, shots: List[Example], **kwargs
    ) -> (str, Dict[str, Any]):

        # prepare the shots
        completed_prompts = []
        for shot in shots:
            _, meta = self.process_example(shot, simulate=True, **kwargs)
            completed_prompts.append(meta["completed_prompt"])
        completed_prompt = "\n\n".join(completed_prompts)

        # process the example
        return self.process_example(eg, flow=completed_prompt, **kwargs)

    def process_example(
        self, eg: Example, simulate: bool = False, flow: str = ""
    ) -> (List[str], Dict[str, Any]):
        meta = {}
        if len(self.templates) == 0:
            raise ValueError("No template was provided.")

        # run each reasoning step
        answers = []
        for key, prompt_template in self.templates.items():
            rich.print(
                f">>[green] Running: {key}, simulate={simulate} : {prompt_template}"
            )
            prompt = prompt_template(eg)
            if simulate:
                prompt_completion = prompt_template.simulate_completion(
                    eg, **prompt_template.completion_config
                )
            else:
                prompt_completion = self._get_prompt_completion(
                    f"{flow}{prompt}", **prompt_template.completion_config
                )
            meta[prompt_template.name] = prompt_completion
            rich.print(f">>[magenta] Got: {key} : {prompt_completion}")
            flow += f"{prompt}{prompt_completion}"
            rich.print(f"[white]{flow}")

            # infer the answer
            answer = prompt_template.infer_answer(
                prompt_completion, eg=eg, pre_answer=flow[: -len(prompt_completion)]
            )  # noqa
            answers.append(answer)

        meta["completed_prompt"] = flow
        return answers, meta

    def _get_prompt_completion(self, prompt, **kwargs) -> str:
        self.throttle()

        # arguments
        engine_args = self.get_engine_args(**kwargs)
        max_tokens = engine_args["max_tokens"]

        # add the price of the prompt
        self.total_cost += self.estimate_price(prompt)

        # query the API
        is_cached = False
        if self.is_dryrun:
            n_tokens = int(0.5 * max_tokens)  # expected number of tokens
            rgn = np.random.RandomState(0)
            tokens = rgn.randint(0, self.tokenizer.vocab_size, size=(n_tokens,))
            tokens = tokens.tolist()
            completion = self.tokenizer.decode(tokens)
        else:
            response, is_cached = self.cache(
                openai.Completion.create,
                engine=self.engine,
                prompt=prompt,
                **engine_args,
            )
            if len(response["choices"]) != 1:
                raise NotImplementedError("Pricing is only implemented for one result")
            completion = response["choices"][0]["text"]

        # keep track of the calls (except when using cached results)
        if not is_cached:
            self.timestamp()
            # add the price for the completion
            self.total_cost += self.estimate_price(completion)

        completion = completion.split("<|endoftext|>")[0]
        return completion

    def throttle(self):
        """make sure to remains within the end-user rate limits
        of no more than 60 requests per minute"""
        RATE_BASE_DURATION = 60
        last_calls = self.last_calls(RATE_BASE_DURATION)
        if len(last_calls) >= self.max_rate - 1:
            sleep_time = RATE_BASE_DURATION - (time.time() - last_calls[0])
            time.sleep(max(0, sleep_time))

    def estimate_price(self, prompt) -> float:
        """estimate the price of each call"""
        encoded_tokens = self.tokenizer.encode(prompt)
        cost = len(encoded_tokens) * self.price / 1000
        return cost

    def __process_example(
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

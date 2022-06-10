import hashlib
import os
import time
from copy import copy
from pathlib import Path
from typing import Any, Callable
from typing import Dict
from typing import List

import dill
import loguru
import numpy as np
import openai
import rich
from dotenv import load_dotenv
from transformers import GPT2Tokenizer
from datasets.fingerprint import Hasher

from medical_reasoning.models.templates import ChainOfThoughtTemplate
from medical_reasoning.models.templates import MultipleChoiceTemplate
from medical_reasoning.utils.datastruct import Example

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")



def update_hash(obj: Any, hasher: Hasher):
    if isinstance(obj, (set, tuple, list)):
        for el in obj:
            update_hash(el, hasher)
    elif isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda x: x[0]):
            update_hash(k, hasher)
            update_hash(v, hasher)
    else:
        hasher.update(obj)




class CachedFunction(object):
    def __init__(self, cache_dir: os.PathLike):
        self.cache_dir = Path(cache_dir)

    def __call__(self,
                 fn: Callable,
                 *args,
                 **kwargs) -> (Any, bool):

        # save the arguments
        data = copy(kwargs)
        data["__args__"] = list(args)
        data["__fn__"] = fn

        # fingerprint
        hasher = Hasher()
        update_hash(data, hasher)
        fingerprint = hasher.hexdigest()
        filename = f"{fingerprint}.pkl"
        cache_file = self.cache_dir / filename

        if cache_file.exists():
            return dill.load(open(cache_file, "rb")), True
        else:
            result = fn(*args, **kwargs)
            dill.dump(result, open(cache_file, "wb"))
            return result, False

class Reasoner(object):
    def __init__(
            self,
            *,
            engine: str = "text-ada-001",
            prompt_mode="chain_of_thought",
            template: ChainOfThoughtTemplate = ...,
            price: float,
            tokenizer: GPT2Tokenizer,
            max_tokens: int = 256,
            max_rate: float = 60,
            cache_dir: os.PathLike = None,
    ):
        self.engine = engine
        self.template = template
        self.price = price
        self.tokenizer = tokenizer
        self.prompt_mode = prompt_mode
        self.max_tokens = max_tokens
        self.max_rate = max_rate
        self.cache = CachedFunction(cache_dir=cache_dir)

        assert self.prompt_mode in {
            "option_chain_of_thought",
            "chain_of_thought",
            "zero_shot",
        }

        # state
        self.reset_stats()

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
            self, prompt, stop="<|endoftext|>", max_tokens=None
    ) -> str:
        self.throttle()

        if max_tokens is None:
            max_tokens = self.max_tokens

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
                temperature=0,  # todo: increase
                max_tokens=max_tokens,
                # max_tokens=512,
                top_p=1,
                # logprobs=5,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop,
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

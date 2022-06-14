from __future__ import annotations

import os
import time
from collections import OrderedDict
from copy import copy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import openai
import yaml
from dotenv import load_dotenv
from transformers import GPT2Tokenizer

from medical_reasoning import configs
from medical_reasoning.models.cache import CachedFunction
from medical_reasoning.models.templates import PromptTemplate
from medical_reasoning.models.verifiers import Verifier
from medical_reasoning.utils.datastruct import Example
from medical_reasoning.utils.datastruct import Prediction

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_CONFIG_PATH = (
    Path(configs.__file__).parent / "model" / "config" / "default.yaml"
)


def extend_flows(flows: str | List, flows_: List[str]) -> List:
    if isinstance(flows, str):
        return [flows + f_ for f_ in flows_]
    elif isinstance(flows, list):
        return [extend_flows(f, flows_) for f in flows]


def flatten(x: List | Any) -> List | Any:
    if isinstance(x, (list, set, tuple)):
        return [j for i in x for j in flatten(i)]
    else:
        return [x]


class Reasoner(object):
    def __init__(
        self,
        *,
        engine: str = "text-ada-001",
        templates: Dict[str, PromptTemplate],
        verifier: Verifier,
        price: float,
        tokenizer: GPT2Tokenizer,
        max_rate: float = 60,
        cache_dir: os.PathLike = None,
        reset_cache: bool = False,
        config: Optional[Dict] = None,
    ):

        self.engine = engine
        self.templates = OrderedDict(templates)
        self.verifier = verifier
        self.price = price
        self.tokenizer = tokenizer
        self.max_rate = max_rate
        self.cache = CachedFunction(cache_dir=cache_dir, reset_cache=reset_cache)

        if config is None:
            config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, "r").read())
        self.config = config

        # state
        self.reset_stats()

    @property
    def strategy(self) -> str:
        return "+".join([template.description for template in self.templates.values()])

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
        return (
            f"Reasoner("
            f"engine={self.engine}, "
            f"templates={self.templates}, "
            f"verifier={self.verifier}"
            f")"
        )

    def __call__(
        self, eg: Example, shots: List[Example], **kwargs
    ) -> (str, Dict[str, Any]):

        # prepare the shots
        completed_prompts = []
        for shot in shots:
            _, flows = self.process_example(shot, simulate=True, **kwargs)
            completed_prompts.extend(flows)
        completed_prompt = "\n\n".join(completed_prompts)

        # process the example
        return self.process_example(eg, flow=completed_prompt, **kwargs)

    def apply_template(
        self,
        template: PromptTemplate,
        *,
        eg: Example,
        simulate: bool = False,
        flow: str,
        meta: Dict,
    ) -> (List[str], List[str]):
        prompt = template(eg)
        engine_args = self.get_engine_args(**template.completion_config)
        if simulate and template.can_be_simulated:
            # enforce returning only one sample
            engine_args = copy(engine_args)
            engine_args["n"] = 1

            # simulate the completion
            prompt_completion = template.simulate_completion(eg, **engine_args)
            prompt_completions = [prompt_completion]
        else:
            prompt_completions = self.get_prompt_completions(
                f"{flow}{prompt}", **engine_args
            )

        # store the prompt completions
        completion_key = f"{template.name}.completions"
        if completion_key in meta:
            meta[completion_key].extend(prompt_completions)
        else:
            meta[completion_key] = prompt_completions

        # infer the answers
        answers = [
            template.infer_answer(
                prompt_completion, eg=eg, pre_answer=flow[: -len(prompt_completion)]
            )
            for prompt_completion in prompt_completions
        ]

        # store the answers
        answer_key = f"{template.name}.answers"
        if answer_key in meta:
            meta[answer_key].extend(answers)
        else:
            meta[answer_key] = answers

        # extend the flow
        flows_extensions = [
            f"{prompt}{prompt_completion}" for prompt_completion in prompt_completions
        ]
        flows = extend_flows(flow, flows_extensions)
        return flows, answers

    def apply_templates(
        self,
        templates: list[PromptTemplate],
        *,
        flows: str | List,
        **kwargs,
    ) -> (List, List[List[str]]):
        answers = []
        for template_i in templates:
            output_flows_i = []
            answers_i = []
            for flow in flows:
                output_flows_ij, answers_ij = self.apply_template(
                    template_i, flow=flow, **kwargs
                )
                output_flows_i.extend(output_flows_ij)
                answers_i.extend(answers_ij)

            # update the flows and the answers
            flows = output_flows_i
            answers.append(answers_i)

        return flows, answers

    def process_example(
        self, eg: Example, simulate: bool = False, flow: str = ""
    ) -> (Prediction, List[str]):
        meta = {}
        if len(self.templates) == 0:
            raise ValueError("No template was provided.")

        # run each reasoning step
        flows, answers = self.apply_templates(
            templates=list(self.templates.values()),
            flows=[flow],
            eg=eg,
            simulate=simulate,
            meta=meta,
        )

        # infer the answer and returns with the completed flows
        prediction = self.verifier(answers, eg=eg, meta=meta)
        return prediction, flows

    def get_prompt_completions(self, prompt, **kwargs) -> List[str]:
        self.throttle()

        # arguments
        max_tokens = kwargs["max_tokens"]
        n = kwargs["n"]

        # add the price of the prompt
        self.total_cost += self.estimate_price(prompt)

        # query the API
        is_cached = False
        if self.is_dryrun:
            n_tokens = int(0.5 * max_tokens)  # expected number of tokens
            completions = [self._simulate_completion(n_tokens) for _ in range(n)]
        else:
            response, is_cached = self.cache(
                openai.Completion.create,
                engine=self.engine,
                prompt=prompt,
                **kwargs,
            )
            completions = [row["text"] for row in response["choices"]]

        # keep track of the calls (except when using cached results)
        if not is_cached:
            if not self.is_dryrun:
                self.timestamp()
            # add the price for the completion
            max_price = max([self.estimate_price(c) for c in completions])
            self.total_cost += max_price * len(completions)

        completions = self._cleanup_completions(completions)
        return completions

    def _cleanup_completions(self, completions):
        completions = [
            completion.split("<|endoftext|>")[0] for completion in completions
        ]
        return completions

    def _simulate_completion(self, n_tokens):
        rgn = np.random.RandomState(0)
        tokens = rgn.randint(0, self.tokenizer.vocab_size, size=(n_tokens,))
        tokens = tokens.tolist()
        completion = self.tokenizer.decode(tokens)
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

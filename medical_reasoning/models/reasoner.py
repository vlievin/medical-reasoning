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

from loguru import logger
import numpy as np
import omegaconf
import openai
import yaml
from dotenv import load_dotenv
from transformers import GPT2Tokenizer

from medical_reasoning import configs
from medical_reasoning.models.cache import CachedFunction
from medical_reasoning.models.stop import Stop
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
            shot_templates: Optional[Dict[str, PromptTemplate]] = None,
            verifier: Verifier,
            price: float,
            tokenizer: GPT2Tokenizer,
            max_rate: float = 60,
            cache_dir: os.PathLike = None,
            reset_cache: bool = False,
            config: Optional[Dict] = None,
            pre_prompt: Optional[str] = None,
            separator: str = "\n\n",
            stop: Optional[Stop] = None,
    ):

        self.engine = engine
        if shot_templates is None:
            shot_templates = copy(templates)
        self.templates = OrderedDict(templates)
        self.shot_templates = OrderedDict(shot_templates)
        self.verifier = verifier
        self.price = price
        self.tokenizer = tokenizer
        self.max_rate = max_rate
        self.cache = CachedFunction(cache_dir=cache_dir, reset_cache=reset_cache)
        self.pre_prompt = pre_prompt
        self.separator = separator
        self.stop = stop

        if config is None:
            config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, "r").read())
        if isinstance(config, omegaconf.DictConfig):
            config = omegaconf.OmegaConf.to_container(config)
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
        if self.pre_prompt is not None:
            completed_prompts.append(self.pre_prompt)

        shot_templates = list(self.shot_templates.values())
        for shot in shots:
            _, flows = self.process_example(
                shot, simulate=True, templates=shot_templates, **kwargs
            )
            completed_prompts.extend(flows)
        completed_prompt = self.separator.join(completed_prompts)
        if len(completed_prompt) > 0:
            completed_prompt += self.separator

        # process the example
        templates = list(self.templates.values())
        return self.process_example(
            eg, flow=completed_prompt, templates=templates, **kwargs
        )

    def apply_template(
            self,
            template: PromptTemplate,
            *,
            eg: Example,
            simulate: bool = False,
            flow: str,
            meta: Dict,
    ) -> (List[str], List[str], bool):
        prompt = template(eg)
        engine_args = self.get_engine_args(**template.completion_config)
        if simulate:
            # enforce returning only one sample
            engine_args = copy(engine_args)
            engine_args["n"] = 1
            if template.can_be_simulated(eg):
                # simulate the completion
                prompt_completion = template.simulate_completion(eg)
                prompt_completions = [prompt_completion]
            else:
                prompt_completions = self.get_prompt_completions(prompt, **engine_args)
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
        if simulate:
            answers = []
        else:
            answers = [
                template.infer_answer(
                    prompt_completion, eg=eg, pre_answer=flow[: -len(prompt_completion)]
                )
                for prompt_completion in prompt_completions
            ]

        # check if the completion chain can be stopped
        if self.stop is not None:
            stops = [self.stop(c, eg=eg, meta=meta) for c in prompt_completions]
            stop = all(stops)
        else:
            stop = False

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
        return flows, answers, stop

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
            stop_i = []
            for flow in flows:
                output_flows_ij, answers_ij, stop_ij = self.apply_template(
                    template_i, flow=flow, **kwargs
                )
                output_flows_i.extend(output_flows_ij)
                answers_i.extend(answers_ij)
                stop_i.append(stop_ij)

            # update the flows and the answers
            flows = output_flows_i
            answers.append(answers_i)
            if all(stop_i):
                break

        return flows, answers

    def process_example(
            self,
            eg: Example,
            *,
            templates: List[PromptTemplate],
            simulate: bool = False,
            flow: str = "",
    ) -> (Optional[Prediction], List[str]):
        meta = {}
        if len(self.templates) == 0:
            raise ValueError("No template was provided.")

        # run each reasoning step
        flows, answers = self.apply_templates(
            templates=templates,
            flows=[flow],
            eg=eg,
            simulate=simulate,
            meta=meta,
        )

        # infer the answer and returns with the completed flows
        if not simulate:
            prediction = self.verifier(answers, eg=eg, meta=meta)
        else:
            prediction = None
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
                retries=True,
            )
            completions = [row["text"] for row in response["choices"]]

        # keep track of the calls (except when using cached results)
        if not is_cached:
            if not self.is_dryrun:
                self.timestamp()
            # add the price for the completion
            max_price = max([self.estimate_price(c) for c in completions])
            self.total_cost += max_price * len(completions)

        completions = [self._cleanup_completion(c, end_tokens=kwargs['stop'])
                       for c in completions]

        return completions

    def _cleanup_completion(self, completion, end_tokens=None):
        if end_tokens is None:
            end_tokens = ["<|endoftext|>"]
        matched_tokens = [t for t in end_tokens if t in completion]
        if len(matched_tokens) == 0:
            return completion
        first_token = min([completion.index(t) for t in matched_tokens])
        parts = completion.split(first_token)
        if len(parts) > 1:
            logger.warning(
                f"Found generated text after end token: "
                f"{first_token}: {first_token.join(parts[1:])}"
            )
        completion = parts[0]
        return completion

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
        if len(last_calls) > self.max_rate:
            sleep_time = RATE_BASE_DURATION - (time.time() - last_calls[0])
            time.sleep(max(0, sleep_time))

    def estimate_price(self, prompt) -> float:
        """estimate the price of each call"""
        encoded_tokens = self.tokenizer.encode(prompt)
        cost = len(encoded_tokens) * self.price / 1000
        return cost

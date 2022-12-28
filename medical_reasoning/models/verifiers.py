import abc
from collections import Counter
from typing import Dict
from typing import List

import numpy as np

from medical_reasoning.utils.datastruct import Example
from medical_reasoning.utils.datastruct import Prediction
from loguru import logger

class Verifier(object):
    @abc.abstractmethod
    def __call__(
        self, answers: List[List[str]], *, eg: Example, meta: Dict
    ) -> Prediction:
        raise NotImplementedError()


class MajorityVotingVerifier(Verifier):
    def __init__(self, template_idx: int = -1):
        self.template_idx = template_idx

    def __call__(
        self, answers: List[List[str]], *, eg: Example, meta: Dict
    ) -> Prediction:
        # select the answers to process
        answer_candidates = answers[self.template_idx]

        # get the frequencies of answers
        freqs = Counter(answer_candidates)
        meta["answer_frequencies"] = freqs

        # compute the probabilities
        probs = np.zeros((len(eg.option_symbols),))
        prediction_idx_per_sample = []
        for a in answer_candidates:
            try:
                i = eg.option_symbols.index(a)
            except Exception as exc:
                logger.warning(
                    f"Answer {a} not found in option symbols {eg.option_symbols}. "
                    f"Exception: {exc}"
                )
                i = np.random.randint(0, len(eg.option_symbols))

            probs[i] += 1
            prediction_idx_per_sample.append(i)

        probs /= len(answer_candidates)
        pred_str = freqs.most_common(1)[0][0]

        return Prediction(
            prediction_str=pred_str,
            example=eg,
            meta=meta,
            probs=probs.tolist(),
            prediction_idx_per_sample=prediction_idx_per_sample,
        )

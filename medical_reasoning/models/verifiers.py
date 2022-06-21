import abc
from collections import Counter
from typing import Dict
from typing import List

import numpy as np

from medical_reasoning.utils.datastruct import Example
from medical_reasoning.utils.datastruct import Prediction


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
        probs = np.zeros((len(eg.allowed_options),))
        for a in answer_candidates:
            if a is None:
                i = np.random.randint(0, len(eg.allowed_options))
            else:
                i = eg.allowed_options.index(a)
            probs[i] += 1

        probs /= len(answer_candidates)

        # return
        pred_str = freqs.most_common(1)[0][0]
        return Prediction(prediction_str=pred_str,
                          example=eg,
                          meta=meta,
                          probs=probs.tolist(),
                          )

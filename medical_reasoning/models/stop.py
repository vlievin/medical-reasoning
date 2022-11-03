from __future__ import annotations

import abc
from typing import Dict
from typing import List

from medical_reasoning.utils.datastruct import Example


class Stop(object):
    @abc.abstractmethod
    def __call__(
            self, completion: str, *, eg: Example, meta: Dict
    ) -> bool:
        raise NotImplementedError()


class StopIfContainsAnswer(Stop):

    def __init__(self, answer_pattern: str | List[str] = "the answer is", lowercase: bool = True):
        if isinstance(answer_pattern, str):
            answer_pattern = [answer_pattern]
        self.answer_pattern = answer_pattern
        self.lowercase = lowercase

    def __call__(
            self, completion: str, *, eg: Example, meta: Dict
    ) -> bool:
        if self.lowercase:
            completion = completion.lower()

        stop = any([p in completion for p in self.answer_pattern])

        return stop

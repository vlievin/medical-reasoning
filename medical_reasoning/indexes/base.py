from __future__ import annotations

import abc
from typing import List
from typing import Optional

from datasets import DatasetDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel


class SearchResults(BaseModel):
    titles: List[List[str]]
    scores: List[List[float]]
    texts: List[List[str]]


class Index(object):
    def __init__(
        self,
        *,
        corpus: DictConfig | DatasetDict,
        subset: Optional[int] = None,
    ):
        super(Index, self).__init__()
        if isinstance(corpus, DictConfig):
            corpus: DatasetDict = instantiate(corpus)
        self.corpus = corpus
        self.subset = subset

    @abc.abstractmethod
    def __call__(
        self, queries: List[str], query_titles: List[str], *, k: int = 10
    ) -> SearchResults:
        raise NotImplementedError()

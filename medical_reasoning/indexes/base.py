from __future__ import annotations

import abc
from typing import List
from typing import Optional

import datasets
import numpy as np
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets.search import BatchedNearestExamplesResults
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel


class SearchResults(BaseModel):
    indices: List[List[int]]
    titles: List[List[str]]
    scores: List[List[float]]
    texts: List[List[str]]


def cast_array(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return x


class Index(object):
    def __init__(
        self,
        *,
        corpus: DictConfig | DatasetDict,
        subset: Optional[int] = None,
        prepare_corpus: bool = True,
        index_column: str = "id",
    ):
        super(Index, self).__init__()

        # cast and slice the corpus
        if prepare_corpus:
            corpus, subset = self._prepare_corpus(corpus, subset)

        # store attribute
        self.corpus: Dataset = corpus
        self.subset = subset
        self.index_column = index_column

        # validate index
        self._validate_index()

    def _prepare_corpus(self, corpus, subset):
        if isinstance(corpus, DictConfig):
            corpus: DatasetDict | Dataset = instantiate(corpus)
        if isinstance(corpus, DatasetDict):
            assert set(corpus.keys()) == {datasets.Split.TRAIN}
            corpus = corpus["train"]
        if subset:
            if isinstance(subset, bool):
                subset = 1000
            subset_indices = list(range(subset))
            corpus = corpus.select(subset_indices[:subset])
        return corpus, subset

    def _validate_index(self):
        # check the first element
        first_idx = self.corpus[0][self.index_column]
        if int(first_idx) != 1:
            raise ValueError(
                f"Index column {self.index_column} must start at 1, not `{first_idx}`"
            )

        # check if the index is contiguous
        bs = 100
        nmax = 1000
        for i in range(0, nmax, bs):
            batch = self.corpus[i : i + bs]
            ids = batch[self.index_column]
            if not all(
                [int(ids[j + 1]) == int(ids[j]) + 1 for j in range(len(ids) - 1)]
            ):
                raise ValueError(
                    "Index is not contiguous. "
                    "Please make sure that the index column is named 'id' "
                    "and that the index is contiguous."
                )

    @abc.abstractmethod
    def __call__(
        self, queries: List[str], aux_queries: Optional[List[str]], *, k: int = 10
    ) -> SearchResults:
        raise NotImplementedError()

    @staticmethod
    def _format_batch_results(
        batch_results: BatchedNearestExamplesResults,
    ) -> SearchResults:
        # format the results
        egs = batch_results.total_examples
        n_egs = len(egs)
        scores = [cast_array(score) for score in batch_results.total_scores]
        return SearchResults(
            scores=scores,
            indices=[egs[i]["id"] for i in range(n_egs)],
            titles=[egs[i]["title"] for i in range(n_egs)],
            texts=[egs[i]["text"] for i in range(n_egs)],
        )

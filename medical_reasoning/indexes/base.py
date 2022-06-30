from __future__ import annotations

import abc
from copy import copy
from typing import List, Dict
from typing import Optional

import datasets
import numpy as np
import rich
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets.search import BatchedNearestExamplesResults
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel

from medical_reasoning.indexes.utils.generate_passages import GeneratePassages


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
            index_column: str = "id",
            passage_length: Optional[int] = None,
            passage_stride: Optional[int] = None,
            num_proc: int = 1,
            prepare_corpus: bool = True,
    ):
        super().__init__()

        # cast and slice the corpus
        if prepare_corpus:
            corpus, subset = self._prepare_corpus(corpus, subset, passage_length, passage_stride,
                                                  num_proc)

        # store attribute
        self.corpus: Dataset = corpus
        self.subset = subset
        self.index_column = index_column
        self.passage_length = passage_length
        self.passage_stride = passage_stride
        self.num_proc = num_proc

        # validate index
        self._validate_index()

    @property
    def base_args(self) -> Dict:
        base_arg_names = [
            "corpus",
            "subset",
            "index_column",
            "passage_length",
            "passage_stride",
            "num_proc",
        ]
        args = copy(self.__dict__)
        return {k: v for k, v in args.items() if k in base_arg_names}

    def _prepare_corpus(self, corpus, subset, passage_length: Optional[int] = None,
                        passage_stride: Optional[int] = None, num_proc: int = 1):
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

        if passage_length is not None:
            assert passage_stride is not None

            pipe = GeneratePassages(
                passage_length=passage_length, passage_stride=passage_stride
            )
            corpus = corpus.map(
                pipe,
                batched=True,
                num_proc=num_proc,
                desc="Generating passages",
                batch_size=5,
            )
            # add `id` column
            if 'id' in corpus.column_names:
                corpus = corpus.remove_columns(['id'])
            corpus = corpus.add_column("id", list(range(1, len(corpus) + 1)))

        return corpus, subset

    def _validate_index(self):
        if not {"id", "text", "title"} <= set(self.corpus.column_names):
            raise ValueError(
                f"The corpus must contain at least the columns 'id', 'text' and 'title'."
                f"Found {self.corpus.column_names}"
            )


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
            batch = self.corpus[i: i + bs]
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

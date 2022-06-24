from __future__ import annotations

import abc
from typing import List
from typing import Optional

import rich
import torch
from datasets import DatasetDict
from omegaconf import DictConfig
from transformers import DPRQuestionEncoder
from transformers import DPRQuestionEncoderTokenizer

from medical_reasoning.indexes.base import Index
from medical_reasoning.indexes.base import SearchResults


class DprIndex(Index):
    def __init__(
        self,
        *,
        corpus: DictConfig | DatasetDict,
        subset: Optional[int] = None,
        hf_model: str = "facebook/dpr-question_encoder-single-nq-base",
    ):
        super(Index, self).__init__(corpus=corpus, subset=subset)
        rich.print(self.corpus)

        self.encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )

    @torch.no_grad()
    def encode(self, questions: List[str]):
        batch = self.tokenizer(questions, return_tensors="pt")
        output = self.encoder(**batch)
        return output

    @abc.abstractmethod
    def __call__(
        self, queries: List[str], query_titles: List[str], *, k: int = 10
    ) -> SearchResults:
        # encode the queries
        queries_with_titles = [f"{q} {qt}" for q, qt in zip(queries, query_titles)]
        vecs = self.encode(queries_with_titles)

        # retrieve the top k nearest examples
        dset = self.corpus["train"]
        scores, retrieved_examples = dset.get_nearest_examples("embeddings", vecs, k=k)
        rich.print(retrieved_examples)
        raise NotImplementedError()

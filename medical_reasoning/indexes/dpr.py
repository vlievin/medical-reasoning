from __future__ import annotations

from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import rich
import torch
from datasets import DatasetDict
from omegaconf import DictConfig
from transformers import DPRQuestionEncoder
from transformers import DPRQuestionEncoderTokenizer
from transformers.models.dpr.modeling_dpr import DPRQuestionEncoderOutput

from medical_reasoning.indexes.base import Index
from medical_reasoning.indexes.base import SearchResults


class DprIndex(Index):
    def __init__(
        self,
        *,
        corpus: DictConfig | DatasetDict,
        subset: Optional[int] = None,
        prepare_corpus: bool = True,
        hf_model: str = "facebook/dpr-question_encoder-single-nq-base",
    ):
        super(DprIndex, self).__init__(
            corpus=corpus, subset=subset, prepare_corpus=prepare_corpus
        )
        self.encoder = DPRQuestionEncoder.from_pretrained(hf_model)
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(hf_model)

    def encode(self, questions: List[str]) -> torch.Tensor:
        batch = self.tokenizer(questions, return_tensors="pt", padding=True)
        output: DPRQuestionEncoderOutput = self.encoder(**batch)
        return output.pooler_output

    @torch.no_grad()
    def __call__(self, queries: List[str], *, k: int = 10) -> SearchResults:
        # encode the queries
        vecs = self.encode(queries)

        # retrieve the top k nearest examples
        batch_results = self.corpus.get_nearest_examples_batch(
            "embeddings", vecs.numpy(), k=k
        )

        return self._format_batch_results(batch_results)

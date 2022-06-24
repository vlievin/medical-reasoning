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


def cast_array(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return x


class DprIndex(Index):
    def __init__(
        self,
        *,
        corpus: DictConfig | DatasetDict,
        subset: Optional[int] = None,
        hf_model: str = "facebook/dpr-question_encoder-single-nq-base",
    ):
        super(DprIndex, self).__init__(corpus=corpus, subset=subset)
        rich.print(self.corpus)

        self.encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )

    def encode(self, questions: List[str]) -> torch.Tensor:
        batch = self.tokenizer(questions, return_tensors="pt", padding=True)
        output: DPRQuestionEncoderOutput = self.encoder(**batch)
        return output.pooler_output

    @torch.no_grad()
    def __call__(self, queries: List[str], *, k: int = 10) -> SearchResults:
        rich.print(queries)

        # encode the queries
        vecs = self.encode(queries)

        # retrieve the top k nearest examples
        batch_results = self.corpus.get_nearest_examples_batch(
            "embeddings", vecs.numpy(), k=k
        )

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

from __future__ import annotations

import functools
from typing import Dict
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
from datasets import DatasetDict
from omegaconf import DictConfig

from medical_reasoning.indexes.base import Index
from medical_reasoning.indexes.base import SearchResults


def flatten(x):
    if isinstance(x, (list, set, tuple)):
        for y in x:
            yield from flatten(y)
    else:
        yield x


class MultiIndex(Index):
    def __init__(
        self,
        *,
        corpus: DictConfig | DatasetDict,
        subset: Optional[int] = None,
        p: int,
        indexes: Dict[str, functools.partial],
        weights: Dict[str, float],
    ):
        super(MultiIndex, self).__init__(
            corpus=corpus, subset=subset, prepare_corpus=True
        )
        self.p = p
        assert set(indexes.keys()) == set(weights.keys())
        self.indexes = {
            name: index_partial(
                corpus=self.corpus, subset=self.subset, prepare_corpus=False
            )
            for name, index_partial in indexes.items()
        }
        self.weights = weights

    def __call__(self, queries: List[str], *, k: int = 10) -> SearchResults:
        # query the index
        all_results = {
            index_name: index(queries, k=self.p)
            for index_name, index in self.indexes.items()
        }

        # aggregate the results
        scores_by_index = [{} for _ in queries]

        # scan and initialize the scores
        for index_name, results in all_results.items():
            for i, score in enumerate(results.scores):
                indices = results.indices[i]
                for j, idx in enumerate(indices):
                    scores_by_index[i][idx] = 0.0

        # scan and fill the scores
        for i, scores_by_index_i in enumerate(scores_by_index):
            for idx in scores_by_index_i.keys():
                for index_name, results in all_results.items():
                    weight = self.weights[index_name]
                    scores_i = {
                        idx: score
                        for idx, score in zip(results.indices[i], results.scores[i])
                    }
                    min_scores = min(scores_i.values())
                    scores_by_index_i[idx] += weight * scores_i.get(idx, min_scores)

        # sort the results
        scores, indices = [], []
        for i, scores_by_index_i in enumerate(scores_by_index):
            scores_i = sorted(
                scores_by_index_i.items(), key=lambda x: x[1], reverse=True
            )
            scores.append([score for idx, score in scores_i][:k])
            indices.append([idx for idx, score in scores_i][:k])

        # fetch the data and return
        indices_ = [idx - 1 for idx in flatten(indices)]
        rows = self.corpus[indices_]
        output = {}
        for key in ["id", "text", "title"]:
            output[key] = [[None for _ in range(k)] for _ in queries]
            z = 0
            for i, idx_i in enumerate(indices):
                for j, idx in enumerate(idx_i):
                    output[key][i][j] = rows[key][z]
                    if key == "id":
                        assert int(output[key][i][j]) == int(indices[i][j])
                    z += 1

        return SearchResults(
            scores=scores,
            indices=output["id"],
            titles=output["title"],
            texts=output["text"],
        )

from __future__ import annotations

import functools
import types
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List

import numpy as np
import rich
from datasets import Dataset
from datasets import DatasetDict
from datasets import NamedSplit


def flat_iter(x: Any) -> Iterator[Any]:
    if isinstance(x, str):
        yield x
    elif isinstance(x, (list, types.GeneratorType)):
        for y in x:
            yield from flat_iter(y)
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from flat_iter(v)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def iter_column(
    dataset: Dataset, column_name: str, batch_size: int = 1000
) -> Iterator[List[Any]]:
    for i in range(0, dataset.shape[0], batch_size):
        yield dataset[i : i + batch_size][column_name]


def apply_nested(func: Any, x: Any) -> Any:
    if isinstance(x, list):
        return [apply_nested(func, y) for y in x]
    elif isinstance(x, dict):
        return {k: apply_nested(func, v) for k, v in x.items()}
    else:
        return func(x)


def set_precision(x: float, prec: str = ".2f") -> float:
    return float(f"{x:{prec}}")


def summarize(
    values: List[Any], prec=".2f", percentiles: List = None
) -> Dict[str, Any]:
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]
    output = {
        "n": len(values),
    }
    if len(values) > 0:
        output.update(
            {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "percentiles": {f"{p}": np.percentile(values, p) for p in percentiles},
            }
        )

    # reduce precision
    prec_func = functools.partial(set_precision, prec=prec)
    if prec is not None:
        output = apply_nested(prec_func, output)

    return output


def get_documents_stats(
    stream: Iterator[str], percentiles: List = None
) -> Dict[str, Any]:
    total = 0
    total_characters = 0
    total_words = 0
    n_words = []
    for doc in stream:
        total += 1
        total_characters += len(doc)
        total_words += len(doc.split())
        n_words.append(len(doc.split()))

    return {
        "size": total,
        "total_characters": total_characters,
        "total_words": total_words,
        "words": summarize(n_words, percentiles=percentiles),
    }


class DatasetStats(object):
    def __init__(
        self,
        question_column: str = "question",
        options_column: str = "answer",
        reasoning_column: str = "reasoning",
        documents_column: str = "documents",
        percentiles: List = None,
    ):
        self.percentiles = percentiles
        self.methods = {
            question_column: self._get_question_stats,
            options_column: self._get_options_stats,
            reasoning_column: self._get_reasoning_stats,
            documents_column: self._get_documents_stats,
        }

    @functools.singledispatchmethod
    def __call__(self, dataset: Dataset | DatasetDict) -> Dict:
        raise TypeError(f"Unsupported type {type(dataset)}")

    @__call__.register(Dataset)
    def _(self, dataset: Dataset) -> Dict:
        output = {}
        for key, method in self.methods.items():
            if key in dataset.column_names:
                output[str(key)] = method(key, dataset, percentiles=self.percentiles)
        return output

    @__call__.register(DatasetDict)
    def _(self, dataset: DatasetDict) -> Dict:
        return {str(split): self(dset) for split, dset in dataset.items()}

    @staticmethod
    def _get_question_stats(key: str, dataset: Dataset, **kwargs) -> Dict:
        questions = iter_column(dataset, key)
        return get_documents_stats(flat_iter(questions), **kwargs)

    @staticmethod
    def _get_options_stats(key: str, dataset: Dataset, **kwargs) -> Dict:
        options = iter_column(dataset, key)
        return get_documents_stats(flat_iter(options), **kwargs)

    @staticmethod
    def _get_reasoning_stats(key: str, dataset: Dataset, **kwargs) -> Dict:
        reasonings = iter_column(dataset, key)
        return get_documents_stats(flat_iter(reasonings), **kwargs)

    @staticmethod
    def _get_documents_stats(key: str, dataset: Dataset, **kwargs) -> Dict:
        documents = iter_column(dataset, key)
        return get_documents_stats(flat_iter(documents), **kwargs)

from __future__ import annotations

import string
from os import PathLike
from typing import List
from typing import Optional

import datasets
import numpy as np
from datasets import Dataset
from datasets import DatasetDict

from medical_reasoning.datasets.builders import medqa
from medical_reasoning.datasets.formatters import HeadQAFormatter
from medical_reasoning.datasets.formatters import MedMCQAFormatter
from medical_reasoning.datasets.formatters import PubMedQAFormatter

QA_DATASETS = {
    "medqa_us": (medqa.__file__, "us"),
    "medqa_tw.yaml": (medqa.__file__, "tw"),
    "pubmedqa": ("pubmed_qa", "pqa_labeled"),
    "headqa": ("head_qa", "en"),
    "medmcqa": ("medmcqa", None),
}

QA_FORMATTERS = {
    "pubmedqa": PubMedQAFormatter,
    "headqa": HeadQAFormatter,
    "medmcqa": MedMCQAFormatter,
}

REQUIRED_COLUMNS = ["question", "options", "answer"]


class DatasetBuilder(object):
    def __init__(
        self,
        *,
        name: str,
        cache_dir: Optional[PathLike] = None,
        splits: Optional[List[datasets.Split]] = None,
        subset: Optional[int] = None,
        options: Optional[List] = None,
        **kwargs,
    ):
        if isinstance(splits, str):
            splits = [splits]
        self.name = name
        self.cache_dir = cache_dir
        self.splits = splits
        self.subset = subset
        self.kwargs = kwargs
        self.options = options

    def __call__(self, *args, **kwargs) -> DatasetDict:
        dset_args = QA_DATASETS[self.name]
        dataset = datasets.load_dataset(
            *dset_args, cache_dir=self.cache_dir, **self.kwargs
        )

        # format the data
        if self.name in QA_FORMATTERS:
            formatter = QA_FORMATTERS[self.name]()
            dataset = formatter(dataset)

        # infer the options if they are not provided
        if self.options is None:
            n_options = set.union(
                *[
                    set([len(o) for o in dset["options"]])
                    for split, dset in dataset.items()
                ]
            )
            if len(n_options) != 1:
                raise ValueError(
                    f"All datasets must have the same number of options. "
                    f"Got {n_options}"
                )
            n_options = list(n_options)[0]
            self.options = string.ascii_uppercase[:n_options]

        # validate the data
        for split, dset in dataset.items():
            if not self._validate_data(dset):
                raise ValueError(f"Invalid dataset for split: {split}")

        # filter the splits
        if self.splits is not None:
            dataset = DatasetDict(
                {split: dset for split, dset in dataset.items() if split in self.splits}
            )

        # sub-sample the dataset
        if self.subset is not None:
            new_dataset = {}
            np.random.RandomState(0)
            for split, dset in dataset.items():
                n = min(len(dset), self.subset)
                indices = np.random.choice(len(dset), size=n, replace=False)
                new_dataset[split] = dset.select(indices)
            dataset = DatasetDict(new_dataset)

        return dataset

    def _validate_data(self, dset: Dataset) -> bool:
        if not set(dset.column_names) >= set(REQUIRED_COLUMNS):
            missing_columns = set(REQUIRED_COLUMNS) - set(dset.column_names)
            raise ValueError(
                f"Invalid dataset: {self.name}: missing columns: {missing_columns}. "
                f"Found: {dset.column_names}"
            )

        if len(dset) == 0:
            raise ValueError(f"Invalid dataset: {self.name}: empty dataset")

        for i in range(len(dset))[:100]:
            row = dset[i]
            answer = row["answer"]
            if not isinstance(answer, int):
                raise ValueError(
                    f"Invalid dataset: {self.name}: answer must be an integer"
                )

            # check the number of options
            n_options = len(row["options"])
            if n_options != len(self.options):
                raise ValueError(
                    f"Invalid dataset: {self.name}: incorrect number of options. "
                    f"Expected: {len(self.options)}, Found: {n_options}"
                )

            # check the answer idx
            if (answer < 0 and answer != -1) or answer >= n_options:
                raise ValueError(
                    f"Invalid dataset: {self.name}: answer must be "
                    f"-1 or in range [0, {len(row['options'])}). "
                    f"Got {answer}"
                )

        return True

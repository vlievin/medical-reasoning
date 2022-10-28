from __future__ import annotations

import string
from os import PathLike
from typing import List
from typing import Optional

import datasets
import numpy as np
import rich
from datasets import Dataset
from datasets import DatasetDict

from medical_reasoning.datasets.builders import medqa
from medical_reasoning.datasets.builders import mmlu_usmle
from medical_reasoning.datasets.builders import pubmedqa
from medical_reasoning.datasets.formatters import HeadQAFormatter
from medical_reasoning.datasets.formatters import MedMCQAFormatter

QA_DATASETS = {
    "medqa_us": (medqa.__file__, "us"),
    "medqa_tw.yaml": (medqa.__file__, "tw"),
    "pubmedqa": (pubmedqa.__file__, "pqa-l"),
    "headqa": ("head_qa", "en"),
    "medmcqa": ("medmcqa", None),
    "mmlu_usmle": (mmlu_usmle.__file__, None),
}

QA_FORMATTERS = {
    "headqa": HeadQAFormatter,
    "medmcqa": MedMCQAFormatter,
}

REQUIRED_COLUMNS = ["question", "options", "answer_idx", "reasoning", "uid"]


class DatasetBuilder(object):
    def __init__(
        self,
        *,
        name: str,
        cache_dir: Optional[PathLike] = None,
        splits: Optional[List[datasets.Split]] = None,
        subset: Optional[int] = None,
        options: Optional[List] = None,
        is_final: bool = False,
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
        self.is_final = is_final

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
            self.options = [c for c in string.ascii_uppercase[:n_options]]

        # validate the data
        for split, dset in dataset.items():
            if not self._validate_data(dset):
                raise ValueError(f"Invalid dataset for split: {split}")

        # check unicity of the uid
        uids = [uid for split, dset in dataset.items() for uid in dset["uid"]]
        if len(uids) != len(set(uids)):
            raise ValueError(f"Duplicate uid found in dataset: {self.name}")

        # filter the splits
        if self.splits is not None:
            dataset = DatasetDict(
                {split: dset for split, dset in dataset.items() if split in self.splits}
            )

        # sub-sample the dataset
        if self.subset is not None:
            new_dataset = {}
            for split, dset in dataset.items():
                rgn = np.random.RandomState(0)
                indices = list(range(len(dset)))
                rgn.shuffle(indices)
                indices = indices[: self.subset]
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
            answer_idx = row["answer_idx"]
            if not isinstance(answer_idx, int):
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
            if (answer_idx < 0 and answer_idx != -1) or answer_idx >= n_options:
                raise ValueError(
                    f"Invalid dataset: {self.name}: answer must be "
                    f"-1 or in range [0, {len(row['options'])}). "
                    f"Got {answer_idx}"
                )

        return True

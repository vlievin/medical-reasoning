from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import omegaconf
import rich
from datasets import Dataset
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from medical_reasoning.datasets import DatasetBuilder
from medical_reasoning.datasets.stats import DatasetStats
from medical_reasoning.indexes.base import Index
from medical_reasoning.utils.datastruct import Example
from medical_reasoning.utils.datastruct import permute_eg


class Preprocessing(TorchDataset):
    """Handle the preprocessing on the data, including sampling documents and making the `shots`."""

    def __init__(
        self,
        dataset: Dataset,
        config: DictConfig,
        option_symbols: List[str],
        use_index: bool,
        permute_options: bool = False,
        strip_reasoning: bool = False,
    ):
        # store the attributes
        self.dataset = dataset
        self.config = config
        omegaconf.OmegaConf.resolve(self.config)
        self.option_symbols = option_symbols
        self.use_index = use_index
        self._is_instantiated = False
        self.permute_options = permute_options
        self.strip_reasoning = strip_reasoning

    def __getitem__(self, item) -> Tuple[int, Example, List[Example]]:
        # dynamically instantiate the dataset if not yet done
        if not self._is_instantiated:
            self._instantiate()

        # get the row of data, validate and potentially sample documents
        eg = self.make_eg(
            self.dataset[item],
            self.option_symbols,
            seed=item,
        )

        # samples the shots
        shots = self.make_shots_egs(
            self.shots_dataset,
            item,
            self.option_symbols,
        )

        return (item, eg, shots)

    def __len__(self):
        return len(self.dataset)

    def _instantiate(self):
        # initialize the dataset used for the shots
        self.shots_dataset = self.make_shots_dataset(self.config)
        # setup the index
        if self.use_index:
            self.index: Optional[Index] = instantiate(self.config.index)
        else:
            self.index = None

        self._is_instantiated = True

    def __getstate__(self):
        state = copy(self.__dict__)
        state.pop("index", None)
        state.pop("shots_dataset", None)
        state["_is_instantiated"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._is_instantiated = False

    def make_shots_egs(
        self,
        shots_dataset: Dataset,
        row_idx: int,
        option_symbols: List,
    ) -> List[Example]:
        """Sample a bunch of Examples from the Shots dataset."""
        shots = []
        if self.config.shots > 0:
            rgn = np.random.RandomState(row_idx)
            shots_indices = rgn.choice(
                list(range(len(shots_dataset))), size=self.config.shots, replace=False
            )
            for j in shots_indices:
                shots.append(
                    self.make_eg(
                        shots_dataset[int(j)],
                        option_symbols,
                        seed=j,
                    )
                )

        return shots

    def make_eg(
        self,
        row: Dict[str, Any],
        option_symbols: List,
        *,
        seed: int,
    ) -> Example:
        """Make an example from a row of data. Potentially sample the index."""
        if self.strip_reasoning:
            row = row.copy()
            row["reasoning"] = None
        eg = Example(**row, option_symbols=option_symbols)

        # potentially permute
        if self.permute_options:
            eg = permute_eg(eg, seed=seed)

        if len(eg.documents) == 0 and self.index is not None:
            eg = self.sample_documents(eg)

        return eg

    @staticmethod
    def make_shots_dataset(config, percentiles=None) -> Optional[Dataset]:
        """Build the dataset used to draw shots from."""
        if config.shots == 0:
            return None

        if percentiles is None:
            percentiles = [50, 90]

        shots_builder: DatasetBuilder = instantiate(
            config.dataset,
            splits="train",
            subset=None,
        )
        shots_dataset = shots_builder()
        shots_dataset = shots_dataset["train"]
        stats = DatasetStats(percentiles=percentiles)
        shots_stats = stats(shots_dataset)
        # take the training split and use only the reasonings in percentiles [50, 95]
        min_length = int(
            shots_stats["reasoning"]["words"]["percentiles"][str(percentiles[0])]
        )
        max_length = int(
            shots_stats["reasoning"]["words"]["percentiles"][str(percentiles[1])]
        )
        pipe = FilterByLength("reasoning", min_length=min_length, max_length=max_length)
        shots_dataset = shots_dataset.filter(
            pipe,
            num_proc=4,
            desc=f"Filtering shots dataset lengths: [{min_length}-{max_length}]",
        )
        shots_stats = stats(shots_dataset)
        rich.print(shots_stats)
        json.dump(shots_stats, Path("shots_stats.json").open("w"), indent=2)
        return shots_dataset

    def sample_documents(self, eg: Example):
        """Sample the documents for a given example."""
        if eg.question_clean is not None:
            base_query = eg.question_clean
        else:
            base_query = eg.question

        queries = [f"{base_query} {opt}" for opt in eg.options]
        results = self.index(queries, aux_queries=eg.options, k=self.config.n_docs)
        documents = []
        if len(results.texts) != len(results.titles):
            raise ValueError("text and title must be of the same length")

        for x, y in zip(results.texts, results.titles):
            if len(x) != len(y):
                raise ValueError("text and title must be of the same number of results")
            for xx, yy in zip(x, y):
                yy = yy.strip('"')
                documents.append(f'{yy}. "{xx}"')

        return eg.copy(update={"documents": documents})


class FilterByLength(object):
    def __init__(
        self,
        key: str,
        *,
        min_length: int = 0,
        max_length: int = None,
        split: bool = True,
    ):
        self.key = key
        self.min_length = min_length
        self.max_length = max_length
        self.split = split

    def __call__(self, row: Dict) -> bool:
        x = row[self.key]
        if self.split:
            x = x.split()
        if self.min_length is not None and len(x) < self.min_length:
            return False
        if self.max_length is not None and len(x) > self.max_length:
            return False
        return True

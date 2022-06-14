import json
import logging
import os
import socket
import warnings
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import datasets
import hydra
import numpy as np
import rich
from datasets import Dataset
from elasticsearch.exceptions import ElasticsearchWarning
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from rich.table import Table
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from slugify import slugify
from tqdm import tqdm

from medical_reasoning.datasets import DatasetBuilder
from medical_reasoning.datasets.stats import DatasetStats
from medical_reasoning.indexes import ElasticsearchIndex
from medical_reasoning.models import Reasoner
from medical_reasoning.utils.config import print_config
from medical_reasoning.utils.datastruct import Example
from medical_reasoning.utils.datastruct import Prediction

SEPARATOR = "-" * 80 + "\n"

OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)
OmegaConf.register_new_resolver("hostname", socket.gethostname)
OmegaConf.register_new_resolver("shorten", lambda x, y: str(slugify(x))[: int(y)])

warnings.filterwarnings(
    action="ignore",
    category=ElasticsearchWarning,
)


hydra


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


@hydra.main(
    config_path="configs/",
    config_name="config.yaml",
    version_base="1.2",
)
def run(config: DictConfig) -> None:
    hydra_config = HydraConfig().get()
    if config.disable_caching:
        datasets.disable_caching()
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    datasets.logging.set_verbosity(datasets.logging.ERROR)

    # write config to file and display it
    with open("config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(config))
    if config.print_config:
        print_config(config)

    # setup the result files
    work_dir = config.sys.work_dir
    data_file = Path("data.json")
    if hydra_config.mode == RunMode.MULTIRUN:
        result_file = Path(work_dir) / hydra_config.sweep.dir / "results.jsonl"
    else:
        result_file = Path(work_dir) / hydra_config.run.dir / "results.jsonl"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.parent.mkdir(parents=True, exist_ok=True)

    # initialize the dataset
    builder: DatasetBuilder = instantiate(config.dataset)
    dataset = builder()
    allowed_options = builder.options
    logger.info(f"Allowed options: {', '.join(allowed_options)}")
    rich.print(f"Dataset:\n{dataset}")
    dataset_stats = DatasetStats()(dataset)
    rich.print(dataset_stats)
    json.dump(dataset_stats, Path("dataset_stats.json").open("w"), indent=2)

    # initialize the dataset used for the shots
    shots_dataset = make_shots_dataset(config)

    # setup the index
    if config.n_docs > 0:
        index: Optional[ElasticsearchIndex] = instantiate(config.index)
    else:
        index = None

    # setting up OpenAI API
    model: Reasoner = instantiate(config.model)

    output_dir = Path(os.getcwd()) / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Logging to {output_dir}")
    splits = list(dataset.keys())
    split_info = [f"{split} ({len(dataset[split])})" for split in splits]
    logger.info(f"Found splits: {', '.join(split_info)}")
    for split in splits:
        dset = dataset[split]
        labels = []
        preds = []
        locators = []
        indices = list(range(len(dset)))
        rgn = np.random.RandomState(0)
        rgn.shuffle(indices)
        for i, row_idx in (
            pbar := tqdm(enumerate(indices), unit="question", total=len(indices))
        ) :
            # get the row of data, validate and potentially sample documents
            eg = make_eg(dset[row_idx], allowed_options, index=index, config=config)

            # samples the shots
            shots = make_shots_egs(
                shots_dataset, row_idx, allowed_options, index=index, config=config
            )

            # process the Example with the model
            pred, flows = model(eg, shots=shots)

            # update the trackers
            labels.append(eg.answer_idx)
            preds.append(pred.idx)

            # log the progress
            f1 = f1_score(labels, preds, average="macro")
            acc = accuracy_score(labels, preds)
            pbar.set_description(
                f"({split}) Acc: {acc:.2%} F1: {f1:.2%} "
                f"({model.n_calls} calls, ${model.total_cost:.2f})"
            )

            # write the result to file
            q_locator = f"{builder.name}_{eg.uid}"
            locators.append(q_locator)
            output_str = format_prediction(eg, pred, q_locator, flows=flows)
            with open(output_dir / f"{q_locator}_{pred.outcome}.txt", "w") as f:
                f.write(output_str)

        # register the results for the whole split
        split_results = {
            "dataset": builder.name,
            "n_samples": len(dset),
            "split": str(split),
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
            "engine": model.engine,
            "strategy": model.strategy,
            "shots": int(config.shots),
            "n_docs": int(config.n_docs),
            "cost": float(model.total_cost),
            "calls": int(model.n_calls),
        }

        # write data
        with open(data_file.as_posix(), "w") as f:
            f.write(
                json.dumps(
                    {
                        **split_results,
                        "labels": labels,
                        "predictions": preds,
                        "locators": locators,
                    }
                )
            )

        # write all results to a shared file
        with open(result_file.as_posix(), "a+") as f:
            f.write(f"{json.dumps(split_results)}\n")

        # print results
        with open(result_file.as_posix(), "r") as f:
            all_results = [json.loads(line) for line in f.readlines()]
        table = format_results_as_table(all_results)
        rich.print(table)
        rich.print(f">> Logged to {output_dir}")

    logger.info(f">> Logged to {output_dir}")


def make_shots_egs(
    shots_dataset: Dataset,
    row_idx: int,
    allowed_options: List,
    *,
    index: Optional[ElasticsearchIndex],
    config: DictConfig,
) -> List[Example]:
    """Sample a bunch of Examples from the Shots dataset."""
    shots = []
    if config.shots > 0:
        rgn = np.random.RandomState(row_idx)
        shots_indices = rgn.choice(
            list(range(len(shots_dataset))), size=config.shots, replace=False
        )
        for j in shots_indices:
            shots.append(
                make_eg(
                    shots_dataset[int(j)], allowed_options, index=index, config=config
                )
            )

    return shots


def make_eg(
    row: Dict[str, Any],
    allowed_options: List,
    *,
    index: Optional[ElasticsearchIndex],
    config: DictConfig,
) -> Example:
    """Make an example from a row of data. Potentially sample the index."""
    eg = Example(**row, allowed_options=allowed_options)
    if len(eg.documents) == 0 and index is not None:
        eg = sample_documents(eg, index=index, config=config)
    return eg


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
    rich.print(f"Shots Dataset:\n{shots_dataset}")
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


def sample_documents(eg: Example, *, index: ElasticsearchIndex, config: DictConfig):
    """Sample the documents for a given example."""
    if eg.question_clean is not None:
        base_query = eg.question_clean
    else:
        base_query = eg.question

    queries = [f"{o} {base_query}" for o in eg.options]
    results = index(queries, eg.options, k=config.n_docs)
    documents = []
    if len(results["text"]) != len(results["title"]):
        raise ValueError("text and title must be of the same length")

    for x, y in zip(results["text"], results["title"]):
        if len(x) != len(y):
            raise ValueError("text and title must be of the same number of results")
        for xx, yy in zip(x, y):
            documents.append(f"Title: {yy}. {xx}")

    return eg.copy(update={"documents": documents})


def format_prediction(
    eg: Example, pred: Prediction, q_locator: str, flows: List[str]
) -> str:
    """Format the prediction for a given example."""
    formatted_options = "\n".join(
        [f"   {eg.allowed_options[i]}) {option}" for i, option in enumerate(eg.options)]
    )
    formatted_flows = ""
    for i, flow in enumerate(flows):
        _sep = "." * len(SEPARATOR)
        formatted_flows += f"[Flow {i + 1}]\n{_sep}\n{flow}\n"
    output_str = (
        f"Outcome: {pred.outcome}\n{SEPARATOR}\n"
        f"Answer: {eg.answer_symbol}: {eg.options[eg.answer_idx]}\n"
        f"Prediction: {pred.label}: {pred.full}\n{SEPARATOR}\n"
        f"Question [{q_locator}]:\n{eg.question}\n\n"
        f"Options:\n{formatted_options}\n{SEPARATOR}\n"
        f"Reasonings: \n\n{formatted_flows}\n"
    )
    return output_str


def format_results_as_table(all_results) -> Table:
    """Format the results of the experiment using `rich.table.Table`."""
    COLUMN_FORMATS = {
        "dataset": "<20",
        "split": "<10",
        "n_samples": "",
        "accuracy": ".2%",
        "f1": ".2%",
        "engine": "<20",
        "strategy": "<32",
        "n_docs": "",
        "shots": "",
        "cost": ".2f",
        "calls": "",
    }

    first_row = all_results[0]
    keys = [c for c in COLUMN_FORMATS.keys() if c in first_row.keys()]
    table = Table(title="Results")
    for key in keys:
        table.add_column(key, justify="center")
    for record in all_results:
        table.add_row(*[f"{record[key]:{COLUMN_FORMATS[key]}}" for key in keys])

    return table


if __name__ == "__main__":
    run()

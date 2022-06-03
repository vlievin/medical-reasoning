import json
import logging
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Optional

import datasets
import hydra
import numpy as np
import rich
from elasticsearch.exceptions import ElasticsearchWarning
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
from rich.table import Table
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm

from medical_reasoning.datasets import DatasetBuilder
from medical_reasoning.indexes import ElasticsearchIndex
from medical_reasoning.models import Reasoner
from medical_reasoning.utils.config import print_config
from medical_reasoning.utils.datastruct import Example
from medical_reasoning.utils.datastruct import Prediction

SEPARATOR = "-" * 80 + "\n"

OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)
OmegaConf.register_new_resolver("hostname", socket.gethostname)

warnings.filterwarnings(
    action="ignore",
    category=ElasticsearchWarning,
)


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
    if config.print_config:
        print_config(config)

    # setup the result file
    work_dir = config.sys.work_dir
    data_file = Path("data.json")
    if hydra_config.mode == RunMode.MULTIRUN:
        result_file = Path(work_dir) / hydra_config.sweep.dir / "results.jsonl"
    else:
        result_file = Path(work_dir) / hydra_config.run.dir / "results.jsonl"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.parent.mkdir(parents=True, exist_ok=True)

    # initialize the data module
    builder: DatasetBuilder = instantiate(config.dataset)
    dataset = builder()
    allowed_options = builder.options
    logger.info(f"Allowed options: {', '.join(allowed_options)}")
    rich.print(dataset)

    # setup the index
    if config.use_documents:
        index: Optional[ElasticsearchIndex] = instantiate(config.index)
    else:
        index = None

    # setting up OpenAI API
    with open_dict(config):
        config.model.template.options = allowed_options
    model: Reasoner = instantiate(config.model)

    output_dir = Path(os.getcwd()) / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Logging to {output_dir}")
    min_duration_per_question = 1.0 / config.rate_limit
    logger.info(
        f"Rate limit: {config.rate_limit}, Min. duration per question: {min_duration_per_question}"
    )
    t0 = time.time()
    splits = list(dataset.keys())
    split_info = [f"{split} ({len(dataset[split])})" for split in splits]
    logger.info(f"Found splits: {', '.join(split_info)}")
    for split in splits:
        dset = dataset[split]
        labels = []
        preds = []
        indices = list(range(len(dset)))
        rgn = np.random.RandomState(0)
        rgn.shuffle(indices)
        for i, row_idx in (
            pbar := tqdm(enumerate(indices), unit="question", total=len(indices))
        ) :

            # get the row of data, validate and potentially sample documents
            row = dset[row_idx]
            eg = Example(**row, allowed_options=allowed_options)
            if len(eg.documents) == 0 and index is not None:
                eg = sample_documents(eg, index=index, config=config)

            # process the Example with the model
            prediction_str, meta = model(
                eg.question, options=eg.options, documents=eg.documents
            )
            pred = Prediction(prediction_str=prediction_str, example=eg, meta=meta)

            # update the trackers
            labels.append(eg.answer_idx)
            preds.append(pred.idx)

            # log the progress
            f1 = f1_score(labels, preds, average="macro")
            acc = accuracy_score(labels, preds)
            pbar.set_description(
                f"({split}) Acc: {acc:.2%} F1: {f1:.2%} {len(eg.documents)} Docs"
            )

            # write the result to file
            q_locator = f"{builder.name}_{split}_{row_idx}"
            output_str = format_prediction(eg, pred, q_locator)
            with open(output_dir / f"{q_locator}_{pred.outcome}.txt", "w") as f:
                f.write(output_str)

            # throttle the API
            if time.time() - t0 < min_duration_per_question:
                t = min_duration_per_question - (time.time() - t0)
                time.sleep(max(t, 0))
            t0 = time.time()

        # register the results for the whole split
        split_results = {
            "dataset": builder.name,
            "n_samples": len(dset),
            "split": str(split),
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
            "engine": model.engine,
            "prompt_mode": model.prompt_mode,
            "identity": str(model.template.identity),
            "grounded": str(config.use_documents),
        }

        # write data
        with open(data_file.as_posix(), "w") as f:
            f.write(
                json.dumps(
                    {
                        **split_results,
                        "labels": labels,
                        "predictions": preds,
                    }
                )
            )

        # write all results to a shared file
        with open(result_file.as_posix(), "a+") as f:
            f.write(f"{json.dumps(split_results)}\n")

        # print results
        with open(result_file.as_posix(), "r") as f:
            all_results = [json.loads(line) for line in f.readlines()]
        table = format_results(all_results)
        rich.print(table)
        rich.print(f">> Logged to {output_dir}")

    logger.info(f">> Logged to {output_dir}")


def sample_documents(eg: Example, *, index: ElasticsearchIndex, config: DictConfig):
    """Sample the documents for a given example."""
    if eg.question_clean is not None:
        base_query = eg.question_clean
    else:
        base_query = eg.question

    queries = [f"{o} {base_query}" for o in eg.options]
    results = index(queries, eg.options, k=config.topk)
    documents = []
    if len(results["text"]) != len(results["title"]):
        raise ValueError("text and title must be of the same length")

    for x, y in zip(results["text"], results["title"]):
        if len(x) != len(y):
            raise ValueError("text and title must be of the same number of results")
        for xx, yy in zip(x, y):
            documents.append(f"Title: {yy}. {xx}")

    return eg.copy(update={"documents": documents})


def format_prediction(eg: Example, pred: Prediction, q_locator: str) -> str:
    formatted_options = "\n".join(
        [f"   {eg.allowed_options[i]}) {option}" for i, option in enumerate(eg.options)]
    )
    output_str = (
        f"Outcome: {pred.outcome}\n{SEPARATOR}\n"
        f"Answer: {eg.answer_symbol}: {eg.options[eg.answer_idx]}\n"
        f"Prediction: {pred.label}: {pred.full}\n{SEPARATOR}\n"
        f"Question [{q_locator}]:\n{eg.question}\n\n"
        f"Options:\n{formatted_options}\n{SEPARATOR}\n"
        f"Reasoning: \n\n{pred.meta['completed_prompt']}\n"
    )
    return output_str


def format_results(all_results) -> Table:
    """Nicely display the results of the experiment using `rich.table.Table`."""
    FMTS = {
        "dataset": "<20",
        "split": "<10",
        "n_samples": "",
        "accuracy": ".2%",
        "f1": ".2%",
        "engine": "<20",
        "prompt_mode": "<20",
        "identity": "<20",
        "grounded": "<16",
    }

    first_row = all_results[0]
    keys = list(first_row.keys())
    table = Table(title="Results")
    for key in keys:
        table.add_column(key, justify="center")
    for record in all_results:
        table.add_row(*[f"{record[key]:{FMTS[key]}}" for key in keys])

    return table


if __name__ == "__main__":
    run()

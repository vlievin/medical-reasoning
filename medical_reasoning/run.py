import json
import logging
import os
import socket
import string
import sys
import time
from pathlib import Path
from typing import Optional

import datasets
import hydra
import numpy as np
import rich
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

SEPARATOR = "-" * 80 + "\n"


OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)
OmegaConf.register_new_resolver("hostname", socket.gethostname)


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

    # initialize the data module
    builder: DatasetBuilder = instantiate(config.dataset)
    dataset = builder()
    allowed_options = builder.options
    logger.info(f"Allowed options: {', '.join(allowed_options)}")
    rich.print(dataset)

    # setup the index
    index: Optional[ElasticsearchIndex] = instantiate(config.index)

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
            row = dset[row_idx]
            question = row["question"]
            options = row["options"]
            answer_idx = row["answer"]
            answer = allowed_options[answer_idx]
            documents = row.get("documents", [])
            if len(documents) == 0 and index is not None:
                documents = sample_documents(index, question, options, config=config)

            rich.print(documents)
            exit()

            prediction, meta = model(question, options=options, documents=documents)
            try:
                prediction_idx = allowed_options.index(prediction)
            except Exception as exc:
                logger.warning(
                    f"Prediction label couldn't be inferred "
                    f"(prediction={prediction}, "
                    f"allowed_options={allowed_options}, "
                    f"answer={meta['answer']} ). "
                    f"Exception: {exc}"
                )
                prediction_idx = -1

            # update the trackers
            labels.append(answer_idx)
            preds.append(prediction_idx)

            # log the progress
            f1 = f1_score(labels, preds, average="macro")
            acc = accuracy_score(labels, preds)
            pbar.set_description(f"({split}) Acc: {acc:.2%} F1: {f1:.2%}")

            # write the result to file
            outcome = "correct" if prediction == answer else "incorrect"
            with open(output_dir / f"{split}_{row_idx}_{outcome}.txt", "w") as f:
                q_locator = f"{builder.name}:{split}:{row_idx}"
                formatted_options = "\n".join(
                    [
                        f"   {allowed_options[i]}) {option}"
                        for i, option in enumerate(options)
                    ]
                )
                pred_str = (
                    options[prediction_idx]
                    if prediction_idx is not None and prediction_idx >= 0
                    else "N/A"
                )
                output_str = (
                    f"Outcome: {outcome}\n{SEPARATOR}\n"
                    f"Answer: {answer}: {options[answer_idx]}\n"
                    f"Prediction: {prediction}: {pred_str}\n{SEPARATOR}\n"
                    f"Question [{q_locator}]:\n{question}\n\n"
                    f"Options:\n{formatted_options}\n{SEPARATOR}\n"
                    f"Reasoning: \n{meta['completed_prompt']}\n"
                )
                f.write(output_str)

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


def sample_documents(index, question, options, *, config):
    queries = [f"{o} {question}" for o in options]
    rich.print(f">> Queries: {queries}")
    qtitles = options
    results = index(queries, qtitles, k=config.topk)
    rich.print(results)
    documents = []
    assert (
        len(
            results["text"],
        )
        == len(results["title"])
    )
    for x, y in zip(results["text"], results["title"]):
        assert len(x) == len(y)
        for xx, yy in zip(x, y):
            documents.append(f"Title: {yy}. {xx}")
    return documents


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

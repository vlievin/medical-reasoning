from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import warnings
from collections import Counter
from pathlib import Path
from typing import List

import datasets
import hydra
import requests
import rich
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
from torch.utils.data import DataLoader
from tqdm import tqdm

import medical_reasoning
from medical_reasoning.datasets import DatasetBuilder
from medical_reasoning.datasets.stats import DatasetStats
from medical_reasoning.models import Reasoner
from medical_reasoning.utils.config import print_config
from medical_reasoning.utils.datastruct import Example
from medical_reasoning.utils.datastruct import Prediction
from medical_reasoning.utils.preprocessing import Preprocessing

SEPARATOR = "-" * 80 + "\n"


def make_info(prompt_style, permute_options, n_docs, shots, strip_reasoning, pre_prompt):
    info_name = prompt_style

    if bool(permute_options):
        info_name += "-permuted"

    if int(n_docs) > 0:
        info_name += f"-{n_docs}docs"

    if int(shots) > 0:
        info_name += f"-{shots}shots"

    if bool(strip_reasoning):
        info_name += "-strip-reasoning"

    if pre_prompt is not None:
        assert isinstance(pre_prompt, str)
        hash = hashlib.sha1(pre_prompt.encode("UTF-8")).hexdigest()
        info_name += f"-pre{hash}"

    return info_name


def load_file_or_url(path: str, key:str=None):

    if path.startswith("http"):
        content = requests.get(path).text
    else:
        path = Path(path)
        if not path.exists():
            path = Path(medical_reasoning.__file__).parent.parent / path
            if not path.exists():
                raise ValueError(f"File {path} does not exist")
        with open(path, "r") as f:
            content = f.read()

    if str(path).endswith(".json"):
        content = json.loads(content)
    elif str(path).endswith(".yaml"):
        content = OmegaConf.load(content)
    elif str(path).endswith(".txt"):
        ...
    else:
        raise ValueError(f"Could not load {path}.")

    if key is not None:
        content = content[key]

    return content


OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)
OmegaConf.register_new_resolver("hostname", socket.gethostname)
OmegaConf.register_new_resolver("shorten", lambda x, y: str(slugify(x))[: int(y)])
OmegaConf.register_new_resolver("make_info", make_info)
OmegaConf.register_new_resolver("load", load_file_or_url)

warnings.filterwarnings(
    action="ignore",
    category=ElasticsearchWarning,
)


def get_first_el(x):
    return x[0]


@hydra.main(
    config_path="configs/",
    config_name="config.yaml",
    version_base="1.2",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    splits = list(dataset.keys())
    option_symbols = builder.options
    dataset_stats = DatasetStats()(dataset)
    json.dump(dataset_stats, Path("dataset_stats.json").open("w"), indent=2)
    rich.print(dataset_stats)

    # initialize the preprocessing object
    use_index = config.n_docs > 0 and "documents" not in dataset[splits[0]].column_names
    preprocessing = {
        split: Preprocessing(
            dataset[split],
            config=config,
            option_symbols=option_symbols,
            use_index=use_index,
            permute_options=config.permute_options,
            strip_reasoning=config.strip_reasoning,
        )
        for split in dataset.keys()
    }

    # setting up OpenAI API
    model: Reasoner = instantiate(config.model)

    output_dir = Path(os.getcwd()) / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Logging to {output_dir}")

    split_info = [f"{split} ({len(dataset[split])})" for split in splits]
    logger.info(f"Found splits: {', '.join(split_info)}")
    for split in splits:
        labels = []
        preds = []
        probs = []
        locators = []
        loader = DataLoader(
            preprocessing[split],
            num_workers=config.num_workers,
            batch_size=1,
            shuffle=config.get("shuffle_loader", False),
            collate_fn=get_first_el,
        )
        for i, (row_idx, eg, shots) in (
                pbar := tqdm(enumerate(loader), unit="question", total=len(loader.dataset))
        ):
            # process the Example with the model
            pred, flows = model(eg, shots=shots)

            # update the trackers
            labels.append(eg.answer_idx)
            preds.append(pred.idx)
            probs.append(pred.probs)

            # log the progress
            f1 = f1_score(labels, preds, average="macro")
            acc = accuracy_score(labels, preds)
            pbar.set_description(
                f"({split}) Acc: {acc:.2%} F1: {f1:.2%} "
                f"({model.n_calls} calls, ${model.total_cost:.2f})"
            )

            # write the result to file
            uid = eg.uid
            if str(split) not in uid:
                uid = f"{split}-{uid}"
            q_locator = f"{builder.name}_{uid}"
            locators.append(q_locator)
            output_str = format_prediction(eg, pred, q_locator, flows=flows)
            fname = f"{q_locator}_{pred.outcome}_{eg.answer_symbol}_{pred.label}.txt"
            fname = fname.replace("/", "")
            with open(output_dir / fname, "w") as f:
                f.write(output_str)

        # replace answers that couldn't be predicted with the most common answer
        preds_freq = Counter(preds).most_common()
        most_common_pred = preds_freq[0][0]
        n_missing = 0
        for i, pred in enumerate(preds):
            if pred < 0 or pred is None:
                preds[i] = most_common_pred
                n_missing += 1
        logger.info(
            f"{n_missing}/{len(preds)} questions could not be predicted, "
            f"filled with {most_common_pred}"
        )

        # register the results for the whole split
        split_results = {
            "dataset": builder.name,
            "n_samples": len(preprocessing[split]),
            "split": str(split),
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
            "engine": model.engine,
            "strategy": model.strategy,
            "info": config.info,
            "cost": float(model.total_cost),
            "calls": int(model.n_calls),
            "n_missing": n_missing,
        }

        # write data

        with open(data_file.as_posix(), "w") as f:
            f.write(
                json.dumps(
                    {
                        **split_results,
                        "labels": labels,
                        "predictions": preds,
                        "probs": probs,
                        "locators": locators,
                        "preds_freq": preds_freq,
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


def format_prediction(
        eg: Example,
        pred: Prediction,
        q_locator: str,
        flows: List[str],
) -> str:
    """Format the prediction for a given example."""
    formatted_options = "\n".join(
        [f"   {eg.option_symbols[i]}) {option}" for i, option in enumerate(eg.options)]
    )
    formatted_flows = ""
    for i, flow in enumerate(flows):
        _sep = "." * len(SEPARATOR)
        formatted_flows += f"\n[Flow {i + 1}]\n{_sep}\n{flow}\n"
    output_str = (
        f"Outcome: {pred.outcome}\n"
        f"{SEPARATOR}\n"
        f"Answer: {eg.answer_symbol}: {eg.options[eg.answer_idx]}\n"
        f"Prediction: {pred.label}: {pred.full}\n"
        f"Probs: {pred.probs}\n"
        f"{SEPARATOR}\n"
        f"Question [{q_locator}]:\n{eg.question}\n\n"
        f"Options:\n{formatted_options}\n"
        f"{SEPARATOR}\n"
        f"Reasoning:\n{formatted_flows}\n"
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
        "info": "",
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

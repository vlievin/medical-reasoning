import logging
import os
import string
import sys
import time
from pathlib import Path

import datasets
import hydra
import numpy as np
import rich
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm

from medical_reasoning.utils.config import print_config


@hydra.main(
    config_path="configs/",
    config_name="config.yaml",
    version_base="1.2",
)
def run(config: DictConfig) -> None:
    logging.getLogger("openai").setLevel(logging.WARNING)
    datasets.logging.set_verbosity(datasets.logging.ERROR)
    print_config(config)

    # initialize the data module
    builder = instantiate(config.dataset)
    dataset = builder()
    rich.print(dataset)

    # setting up OpenAI API
    model = instantiate(config.model)
    rich.print(model)

    output_dir = Path(os.getcwd()) / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    rich.print(f">> Logging to {output_dir}")
    rate_limit = 2
    t0 = time.time()
    splits = ["test"]
    results = {}
    for split in splits:
        dset = dataset[split]
        labels = []
        preds = []
        indices = list(range(len(dset)))
        rgn = np.random.RandomState(0)
        rgn.shuffle(indices)
        for i, row_idx in (
            pbar := tqdm(enumerate(indices), unit=" questions", total=len(indices))
        ) :
            row = dset[row_idx]
            question = row["question"]
            options = row["options"]
            answer_idx = row["answer"]
            answer = string.ascii_uppercase[answer_idx]

            model_answer, meta = model(question, options)
            try:
                model_answer_idx = string.ascii_uppercase.index(model_answer)
            except Exception as exc:
                logger.warning(f"{exc}")
                model_answer_idx = -1

            # update the trackers
            labels.append(answer_idx)
            preds.append(model_answer_idx)

            # log the progress
            f1 = f1_score(labels, preds, average="macro")
            acc = accuracy_score(labels, preds)
            pbar.set_description(f"({split}) Acc: {acc:.2%} F1: {f1:.2%}")

            # write the result to file
            with open(output_dir / f"{split}_{row_idx}.txt", "w") as f:
                outcome = "correct" if model_answer == answer else "incorrect"
                formatted_options = ",".join(
                    [
                        f"{string.ascii_uppercase[i]}) {option}"
                        for i, option in enumerate(options)
                    ]
                )
                output_str = f"""\
                Outcome: {outcome}\n
                Answer: {answer}: {options[answer_idx]}\n
                Prediction: {model_answer}) {options[model_answer_idx]}\n\n\n
                Question ({split}#{row_idx}): {question}\n
                Options: {formatted_options}\n\n
                Reasoning: \n{meta['completed_prompt']}
                """
                f.write(output_str)

            if time.time() - t0 < rate_limit:
                t = rate_limit - (time.time() - t0)
                time.sleep(max(t, 0))
            t0 = time.time()

        results[split] = {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
        }

    for split, accuracy in results.items():
        rich.print(f">> {split}: {accuracy:.3%}")
    rich.print(f">> Logged to {output_dir}")


if __name__ == "__main__":
    run()

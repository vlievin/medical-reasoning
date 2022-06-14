#! /usr/bin/python
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rich
import yaml
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from tqdm import tqdm


def get_first(serie):
    x = serie.values[0]
    assert all(x == y for y in serie.values)
    return x


def majority_vote(serie):
    freqs = Counter(serie.values)
    return freqs.most_common(1)[0][0]


def get_majority_perfs(records, selected_strategies):
    records_ = records[records["strategy"].isin(selected_strategies)]
    expert = records_[["labels", "predictions", "qid"]].groupby("qid")
    expert = expert.agg(
        {
            "labels": get_first,
            "predictions": majority_vote,
        }
    )
    acc = accuracy_score(expert["labels"], expert["predictions"])
    prec = precision_score(expert["labels"], expert["predictions"], average="macro")
    f1 = f1_score(expert["labels"], expert["predictions"], average="macro")
    return {
        "n_experts": len(selected_strategies),
        "accuracy": acc,
        "precision": prec,
        "f1": f1,
    }


formatters = {
    "accuracy": "{:,.1%}".format,
    "f1": "{:,.1%}".format,
    "precision": "{:,.1%}".format,
    "n_tokens": "{:,.0f}".format,
}

if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-p", "--path", help="path to the experiment data", required=True
    )
    parser.add_argument(
        "-n", "--max_perm", help="Maximum permutation budget", default=5
    )
    args = parser.parse_args()
    rich.print(args)
    multirun_path = Path(args.path)
    logger.info(
        f"Loading data from {multirun_path.absolute()}, max. permutations={args.max_perm}"
    )

    # placeholders for the data + parameters
    summary = []
    records = []
    glob_keys = ["strategy"]
    local_keys = ["labels", "predictions"]
    lengths = []

    # get the sorted paths
    sorted_exp_paths = []
    for exp in multirun_path.iterdir():
        if exp.is_dir():
            data_file = exp / "data.json"
            if not data_file.exists():
                continue
            exp_data = json.load(open(data_file, "r"))
            sorted_exp_paths.append((exp, exp_data["accuracy"]))

    # read the data
    sorted_exp_paths = [x[0] for x in sorted(sorted_exp_paths, key=lambda x: -x[1])]
    for exp in sorted_exp_paths:
        data_file = exp / "data.json"
        config_file = exp / "config.yaml"
        exp_data = json.load(open(data_file, "r"))
        cfg = yaml.safe_load(open(config_file, "r"))
        exp_data["strategy"] = exp_data["strategy"].split("+")[0]
        if "prompt_style" in cfg.keys():
            exp_data["strategy"] = f"{cfg['prompt_style']}-{exp_data['strategy']}"
        summary.append(exp_data)

        # read all individual records
        for i in range(len(exp_data["predictions"])):
            record = {k: exp_data[k][i] for k in local_keys}
            record.update({k: v for k, v in exp_data.items() if k in glob_keys})
            record["qid"] = i
            records.append(record)

    summary = pd.DataFrame(summary)
    records = pd.DataFrame(records)

    all_strategies = summary.sort_values("accuracy", ascending=False)[
        "strategy"
    ].values.tolist()
    expert_data = []
    cache_size = 1_000
    max_acc = 0
    best_strategies = None
    for budget in range(1, args.max_perm + 1):
        idx = 0
        total = sum(1 for _ in itertools.permutations(all_strategies, budget))
        try:
            for perm_strategies in (
                pbar := tqdm(
                    itertools.permutations(all_strategies, budget), total=total
                )
            ) :
                output = get_majority_perfs(records, perm_strategies)
                output["idx"] = idx
                if output["accuracy"] > max_acc:
                    max_acc = output["accuracy"]
                    best_strategies = perm_strategies
                    pbar.set_description(
                        f"Budget={budget} (Best: {max_acc:.2f}, n={len(best_strategies)})"
                    )

                # store and truncate
                expert_data.append(output)
                if len(expert_data) > cache_size:
                    expert_data = list(
                        sorted(expert_data, key=lambda x: x["accuracy"], reverse=True)
                    )[:cache_size]

                # increment
                idx += 1
        except KeyboardInterrupt:
            pass

    # write to file
    expert_data = pd.DataFrame(expert_data).sort_values("accuracy", ascending=False)
    with pd.option_context("max_colwidth", 1000):
        expert_data.to_latex(
            buf=multirun_path / "experts-permutations.tex",
            columns=["n_experts", "accuracy", "f1", "precision"],
            formatters=formatters,
            index=False,
        )

    logger.info(f"Best strategies - Accuracy: {max_acc:.2%})")
    for i, strat in enumerate(best_strategies):
        logger.info(f" - {i}: {strat}")
    rich.print(expert_data)

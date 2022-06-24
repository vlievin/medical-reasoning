#! /usr/bin/python
import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path
from typing import Generator
from typing import List

import numpy as np
import pandas as pd
import rich
import seaborn as sns
import torch
import torchmetrics
import transformers
import yaml
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_first(serie):
    x = serie.values[0]
    assert all(x == y for y in serie.values)
    return x


def majority_vote(serie):
    freqs = Counter(serie.values)
    return freqs.most_common(1)[0][0]


def at_idx(it: Generator, i: int):
    return next(itertools.islice(it, i, None))


def mean_reciprocal_rank(preds: List[List], target: List):
    """Computes `Mean Reciprocal Rank`_."""
    ranks = []
    for pred, target in zip(preds, target):
        if target in pred:
            pred_freq = Counter(pred).most_common()
            pred_order = [x[0] for x in pred_freq]
            r = pred_order.index(target) + 1
            ranks.append(1 / r)
        else:
            ranks.append(0.0)

    return torch.mean(torch.tensor(ranks))


formatters = {
    "logp": "{:,.2e}".format,
    "accuracy": "{:,.1%}".format,
    "accuracy@2": "{:,.1%}".format,
    "mrr": "{:,.2f}".format,
    "f1": "{:,.1%}".format,
    "f1@2": "{:,.1%}".format,
    "precision": "{:,.1%}".format,
    "recall": "{:,.1%}".format,
    "n_tokens": "{:,.0f}".format,
}


class ComputeMetrics(object):
    def __init__(
        self, records: pd.DataFrame, n_options=4, eps: float = 1e-1, noise_scale=1e-5
    ):
        self.records = records
        self.n_options = n_options
        self.noise_scale = noise_scale
        assert eps < 1 and eps > 0
        self.eps = eps

        # register the metrics
        self.metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(num_classes=n_options),
                "accuracy@2": torchmetrics.Accuracy(num_classes=n_options, top_k=2),
                "precision": torchmetrics.Precision(
                    average="macro", num_classes=n_options
                ),
                "f1": torchmetrics.F1Score(average="macro", num_classes=n_options),
                "recall": torchmetrics.Recall(average="macro", num_classes=n_options),
            }
        )

    @property
    def metrics_names(self):
        return list(self.metrics.keys()) + ["mrr", "logp"]

    def list_to_log_probs(self, idx, predictions_list, marginal=None):
        # compute the prons
        probs = np.zeros((self.n_options,))
        for p in predictions_list:
            probs[p] += 1

        if marginal is None:
            rgn = np.random.RandomState(idx)
            probs += self.noise_scale * rgn.random(probs.shape)
        else:
            probs += self.noise_scale * marginal

        probs /= len(predictions_list)
        probs = (1 - self.eps) * probs + self.eps / self.n_options
        log_probs = np.log(probs)
        return log_probs

    @staticmethod
    def logp(row):
        log_probs = row["log_probs"]
        target = row["labels"]
        return log_probs[target]

    def __call__(self, args):
        idx, selected_strategies = args
        records_ = self.records[self.records["strategy"].isin(selected_strategies)]
        aggregated_records = records_[["labels", "predictions", "qid"]].groupby("qid")
        aggregated_records = aggregated_records.agg(
            labels=pd.NamedAgg(column="labels", aggfunc=get_first),
            predictions=pd.NamedAgg(column="predictions", aggfunc=majority_vote),
            predictions_list=pd.NamedAgg(column="predictions", aggfunc=list),
        )

        # compute the marginal probabilities
        all_preds = aggregated_records["predictions"].values
        all_freqs = Counter(all_preds)
        marginal = np.zeros((self.n_options,))
        for k, v in all_freqs.items():
            marginal[k] = v / len(all_preds)

        # compute the log probs
        aggregated_records["log_probs"] = [
            self.list_to_log_probs(*x, marginal)
            for x in enumerate(aggregated_records["predictions_list"])
        ]
        aggregated_records["logp"] = aggregated_records.apply(self.logp, axis=1)

        # compute the metrics
        self.metrics.reset()
        logits = torch.stack(
            [torch.from_numpy(x) for x in aggregated_records["log_probs"].values]
        )
        targets = torch.from_numpy(aggregated_records["labels"].values)
        metrics = self.metrics(logits, targets)
        metrics["mrr"] = mean_reciprocal_rank(
            aggregated_records["predictions_list"].values,
            aggregated_records["labels"].values,
        )
        self.metrics.reset()

        return {
            "idx": idx,
            "n_experts": len(selected_strategies),
            **{k: v.item() for k, v in metrics.items()},
            "strategies": selected_strategies,
            "logp": aggregated_records["logp"].sum(),
        }


def keep_topn(expert_data, main_metric):
    if len(expert_data) > cache_size:
        expert_data = list(
            sorted(expert_data, key=lambda x: x[main_metric], reverse=True)
        )[:cache_size]
    return expert_data


def empty_queue(queue, main_metric, max_score, best_output):
    while len(queue) > 0:
        output = queue.pop(0)
        if output[main_metric] > max_score:
            max_score = output[main_metric]
            best_output = output
    return max_score, best_output


def plot_agreement_matrix(summary, output_path):
    N = len(summary)
    X = np.zeros((N, N))
    # diagonal
    for i in range(N):
        # row = summary.iloc[i]
        # acc = accuracy_score(row['labels'], row['predictions'])
        X[i, i] = np.nan

    # top-diagonal: % of aggreement on correct answers
    for i in range(N):
        for j in range(N):
            if j > i:
                row_i = summary.iloc[i]
                row_j = summary.iloc[j]
                assert row_i["labels"] == row_j["labels"]
                # labels = row_i["labels"]
                y_i = row_i["predictions"]
                y_j = row_j["predictions"]
                # filter the correct results
                # y_i = [t for t,l in zip(y_i, labels) if t ==l]
                # y_j = [t for t,l in zip(y_j, labels) if t ==l]
                agg = sum(1 for t_i, t_j in zip(y_i, y_j) if t_i == t_j) / len(y_i)
                # register
                X[i, j] = agg
    # bottom-diagonal: % of aggreement on wrong answers
    for i in range(N):
        for j in range(N):
            if j < i:
                row_i = summary.iloc[i]
                row_j = summary.iloc[j]
                assert row_i["locators"] == row_j["locators"]
                # labels = row_i["labels"]
                y_i = row_i["predictions"]
                y_j = row_j["predictions"]
                # filter the correct results
                # y_i = [t for t,l in zip(y_i, labels) if t != l]
                # y_j = [t for t,l in zip(y_j, labels) if t != l]
                agg = sum(1 for t_i, t_j in zip(y_i, y_j) if t_i == t_j) / len(y_i)
                # register
                X[i, j] = agg

    fig, ax = plt.subplots(figsize=((16, 12)), dpi=300)
    sns.heatmap(
        X,
        annot=False,
        fmt="g",
        ax=ax,
        xticklabels=summary["strategy"],
        yticklabels=summary["strategy"],
        linewidths=0.5,
        # center=0.25,
        cmap=sns.cm.icefire_r,
    )
    plt.tight_layout()
    plt.savefig(Path(output_path) / "expert-agreement.png", dpi=300)
    plt.close()


def strategy2idx(summary: pd.DataFrame, strategy: str):
    strategies = summary["strategy"]
    matches = strategies[strategies == strategy].index
    if len(matches) != 1:
        raise ValueError(f"strategy {strategy} not matched (matches: {matches})")
    return matches[0]


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # arguments
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--path", help="path to the experiment data", required=True)
    parser.add_argument(
        "--metric", help="metric to maximize", default="accuracy+accuracy@2"
    )

    parser.add_argument("--perm_type", help="type of permutations", default="topn")
    parser.add_argument(
        "--topn", help="number of top combinations to display", default=20, type=int
    )
    parser.add_argument("--num_proc", help="number of workers", default=4, type=int)
    parser.add_argument(
        "--n_options", help="number of answer options", default=4, type=int
    )
    parser.add_argument("--sort_summary", help="sort the summary", default=0, type=int)
    parser.add_argument(
        "--max_perm",
        help="Maximum permutation budget",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--min_perm",
        help="Maximum permutation budget",
        default=2,
        type=int,
    )
    args = parser.parse_args()
    metrics = args.metric.split("+")
    main_metric = metrics[0]
    rich.print(args)
    multirun_path = Path(args.path)
    logger.info(
        f"Loading data from {multirun_path.absolute()}, "
        f"max. permutations={args.max_perm}, metric={metrics}"
    )
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    # placeholders for the data + parameters
    summary = []
    records = []
    glob_keys = ["strategy"]
    local_keys = ["labels", "predictions"]
    lengths = []

    for exp in sorted(multirun_path.iterdir(), key=lambda x: x.name):
        if not exp.is_dir() or exp.name[0] == "_":
            continue
        data_file = exp / "data.json"
        config_file = exp / "config.yaml"
        if not data_file.exists() or not config_file.exists():
            continue
        exp_data = json.load(open(data_file, "r"))
        cfg = yaml.safe_load(open(config_file, "r"))
        exp_data["strategy"] = exp_data["strategy"].split("+")[0]
        # if "prompt_style" in cfg.keys():
        #     exp_data["strategy"] = f"{cfg['prompt_style']}-{exp_data['strategy']}"

        # infer the length
        n_tokens = []
        for txt_file in (exp / "output").iterdir():
            if txt_file.name.endswith(".txt"):
                with open(txt_file, "r") as f:
                    content = f.read()
                    reasoning = content.split(exp_data["strategy"])[-1]
                    reasoning = reasoning.split(
                        "Therefore, among A through D, the answer"
                    )[0]
                n_tokens.append(len(tokenizer.encode(reasoning)))
        exp_data["n_tokens"] = np.mean(n_tokens)
        summary.append(exp_data)

        # read all individual records
        qids = exp_data["locators"]
        for i, qid in sorted(enumerate(qids), key=lambda x: x[1]):
            record = {key: exp_data[key][i] for key in local_keys}
            record.update({k: v for k, v in exp_data.items() if k in glob_keys})
            record["qid"] = qid
            records.append(record)

    summary = pd.DataFrame(summary)
    if args.sort_summary:
        summary = summary.sort_values(main_metric, ascending=False)
    summary = summary.reset_index(drop=True)
    # summary.index += 1
    records = pd.DataFrame(records)

    # plot the agreement matrix
    plot_agreement_matrix(summary, args.path)

    # save the summary
    with pd.option_context("max_colwidth", 1000):
        summary.to_latex(
            buf=multirun_path / "summary.tex",
            columns=["strategy", "accuracy", "f1", "n_tokens"],
            float_format="%.2f",
            formatters=formatters,
            index=True,
        )
    rich.print(
        summary[
            ["engine", "strategy", "n_samples", "split", "accuracy", "f1", "n_tokens"]
        ]
    )

    # retrieve the indices:
    with mp.Pool(processes=args.num_proc) as pool:
        expert_data = []
        cache_size = 1_000
        max_score = -math.inf
        best_output = None
        compute_metrics = ComputeMetrics(records, n_options=args.n_options)
        if main_metric not in compute_metrics.metrics_names:
            raise ValueError(
                f"Metric {main_metric} not supported. "
                f"Supported metrics: {compute_metrics.metrics_names}"
            )
        queue = []

        # set arguments
        if args.max_perm > 0:
            perm_range = range(args.min_perm, args.max_perm + 1)
        else:
            perm_range = list(range(args.min_perm, len(summary["strategy"].values) + 1))
        perm_fn = {
            "combinatorial": itertools.combinations,
            "permutation": itertools.permutations,
            "topn": lambda x, y: [x[:y]],
        }[args.perm_type]

        for budget in perm_range:
            total = sum(1 for _ in perm_fn(summary["strategy"].values, budget))
            try:
                permutations = enumerate(perm_fn(summary["strategy"].values, budget))
                if args.num_proc > 1:
                    outputs = pool.imap_unordered(
                        compute_metrics, permutations, chunksize=100
                    )
                else:
                    outputs = map(compute_metrics, permutations)

                for output in (pbar := tqdm(outputs, total=total)) :
                    queue += [output]

                    # store the queue
                    if len(queue) > cache_size:
                        expert_data.extend(queue)
                        max_score, best_output = empty_queue(
                            queue, main_metric, max_score, best_output
                        )
                        metric_desc = ", ".join(
                            [
                                f"{k}: {formatters[k](best_output[k])}"
                                for k in compute_metrics.metrics_names
                            ]
                        )
                        pbar.set_description(
                            f"Budget={budget} ({metric_desc}, n={len(best_output['strategies'])})"
                        )

                        # keep only `cache_size` items
                        expert_data = keep_topn(expert_data, main_metric)

                if len(queue):
                    expert_data.extend(queue)
                    max_score, best_output = empty_queue(
                        queue, main_metric, max_score, best_output
                    )

            except KeyboardInterrupt:
                pass

    # write to file
    expert_data = pd.DataFrame(expert_data)
    expert_data = expert_data.sort_values(metrics, ascending=False)[: args.topn]
    expert_data["strategies"] = expert_data["strategies"].apply(
        lambda x: "+".join([str(strategy2idx(summary, s)) for s in x])
    )
    expert_data = expert_data.reset_index(drop=True)
    expert_data.index += 1
    with pd.option_context("max_colwidth", 1000):
        expert_data.to_latex(
            buf=multirun_path
            / f"experts-permutations-{args.max_perm}-{args.metric}.tex",
            columns=["n_experts", *compute_metrics.metrics_names, "strategies"],
            formatters=formatters,
        )

    logger.info(f"Best strategies - {main_metric}: {max_score:.2%})")
    for i, strat in enumerate(best_output["strategies"]):
        logger.info(f" - {i}: {strat}")
    rich.print(expert_data)

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rich
import transformers
from omegaconf import OmegaConf

from medical_reasoning.run import make_info  # type: ignore


def get_base_arg_parser() -> argparse.PARSER:
    parser = argparse.ArgumentParser(
        description="Generate an html page to visualize the generated texts."
    )
    parser.add_argument("--path", help="path to the experiment data", required=True)
    parser.add_argument(
        "--filter_info", help="keep only run with info matching this", default=None
    )
    parser.add_argument(
        "--filter_zero", help="filter the empty strategy", default=False
    )
    return parser


def load_experiment_data(
    args, compute_length: bool = True
) -> (pd.DataFrame, pd.DataFrame):
    multirun_path = Path(args.path)
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    # placeholders for the data + parameters
    summary = []
    records = []
    glob_keys = ["strategy"]
    local_keys = ["labels", "predictions"]
    for exp in sorted(multirun_path.iterdir(), key=lambda x: x.name):
        if not exp.is_dir() or exp.name[0] == "_":
            continue
        data_file = exp / "data.json"
        config_file = exp / "config.yaml"
        if not data_file.exists() or not config_file.exists():
            continue
        exp_data = json.load(open(data_file, "r"))
        cfg = OmegaConf.load(config_file)
        strategy = exp_data["strategy"].split("+")[0]
        if args.filter_zero and "--" == str(strategy):
            continue
        exp_data["strategy"] = strategy
        exp_data["info"] = cfg.info
        if args.filter_info is not None and args.filter_info != cfg.info:
            continue

        # infer the length
        if compute_length:
            n_tokens = []
            for txt_file in (exp / "output").iterdir():
                if txt_file.name.endswith(".txt"):
                    if strategy == "--":
                        n_tokens_exp = 0
                    else:
                        with open(txt_file, "r") as f:
                            content = f.read()
                            reasoning = content.split(strategy)[-1]
                            reasoning = reasoning.split(
                                "Therefore, among A through D, the answer"
                            )[0]
                            n_tokens_exp = len(tokenizer.encode(reasoning))
                    n_tokens.append(n_tokens_exp)
            exp_data["n_tokens"] = np.mean(n_tokens)
        summary.append(exp_data)

        # read all individual records
        qids = exp_data["locators"]
        for i, qid in sorted(enumerate(qids), key=lambda x: x[1]):
            record = {key: exp_data[key][i] for key in local_keys}
            record.update({k: v for k, v in exp_data.items() if k in glob_keys})
            record["qid"] = qid
            if compute_length:
                record["n_tokens"] = n_tokens[i]
            record["info"] = cfg.info
            records.append(record)

    summary = pd.DataFrame(summary)
    records = pd.DataFrame(records)
    return summary, records

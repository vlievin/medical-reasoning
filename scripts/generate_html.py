#! /usr/bin/python
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import jinja2
import rich
import yaml
from loguru import logger

SEP = 80 * "." + "\n"
ANS_DEL = {"short": "\n\nA: ", "full": "\n\nAnswer: "}

ORDERING = [
    "--",
    "Let's think step by step",
    "Let's think step by step like a medical expert",
    "Let's use step by step inductive reasoning, given the medical nature of the question",
    "Let's differentiate using step by step reasoning like a medical expert",
    "Let's derive the differential diagnosis step by step",
]

locator_re = re.compile(r"\[(\w+)_(\w+)-([\d\w+-]+)\]")

formatted_dset = {"medqa_us": "USMLE", "pubmedqa": "PubMedQA-L", "medmcqa": "MedMCQA"}


def load_data(data_dir: Path) -> (List[str], List[Dict[str, Any]]):
    # placeholders for the data + parameters
    answers = defaultdict(dict)
    questions = dict()
    for exp in data_dir.iterdir():
        if not exp.is_dir():
            continue
        try:
            # data_file = exp / "data.json"
            config_file = exp / "config.yaml"
            # exp_data = json.load(open(data_file, "r"))
            cfg = yaml.safe_load(open(config_file, "r"))
            strategy = cfg["strategy"]["prompt"]
            strategy = strategy.replace("Let’s", "Let's")
            ans_del = ANS_DEL[cfg["prompt_style"]]
            for record in (exp / "output").iterdir():
                if not record.name.endswith(".txt"):
                    continue

                # read and parse the record
                content = open(record, "r").read()
                match = locator_re.search(content)
                dset, split, idx = match.groups()
                dset = formatted_dset[dset]
                content = content.split(SEP)
                flow = content[-1]
                try:
                    question, answer, *_ = flow.split(ans_del)
                except Exception as exc:
                    rich.print(len(flow.split(ans_del)))
                    logger.error(exc)

                # parse the ground truth and the pred
                ground_truth = [
                    c for c in content[0].split("\n") if c.startswith("Answer: ")
                ][0]
                ground_truth = ground_truth.replace("Answer: ", "").replace(":", ")")
                pred = [
                    c for c in content[0].split("\n") if c.startswith("Prediction: ")
                ][0]
                pred = pred.replace("Prediction: ", "").replace(":", ")")
                is_correct = pred == ground_truth
                outcome = "&#9989; " if is_correct else "&#10060; "

                # emphasis the right answer
                question = question.replace(
                    ground_truth, rf"<strong>{ground_truth}</strong>"
                )

                # store
                qid = hashlib.md5(question.encode("utf-8")).hexdigest()
                questions[qid] = {
                    "content": question,
                    "idx": idx,
                    "dataset": dset,
                    "split": split,
                    "qid": qid,
                }
                answers[qid][strategy] = outcome + answer.replace("Let’s", "Let's")
        except Exception as exc:
            logger.error(exc)

    # format
    strategies = list(set.union(*[set(q.keys()) for q in answers.values()]))
    strategies = list(sorted(strategies, key=lambda s: ORDERING.index(s)))
    data = []
    for qid, question in questions.items():
        data.append(
            {
                "question": question,
                "answers": [answers[qid].get(s, "<missing>") for s in strategies],
            }
        )
    return strategies, data


def make_template(args):
    data_dir = Path(args.path)
    output_path = Path(args.fname)
    logger.info(f"Reading data from {data_dir}")

    # load the data
    strategies, data = load_data(data_dir)

    # load the Jinja2 template
    template = load_template("showcase.html")

    # write the template
    template = template.render(
        title="Can large language models reason about medical questions?",
        strategies=strategies,
        data=data,
        ncols=len(strategies),
    )
    with open(output_path, "w") as f:
        f.write(template)


def load_template(fname):
    templateLoader = jinja2.FileSystemLoader(
        searchpath=Path(__file__).parent / "templates"
    )
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template(fname)
    return template


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(
        description="Generate an html page to visualize the generated texts."
    )
    parser.add_argument("--path", help="path to the experiment data", required=True)
    parser.add_argument("--fname", help="output file name", default="main.html")
    args = parser.parse_args()
    make_template(args)

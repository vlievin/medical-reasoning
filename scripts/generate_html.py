#! /usr/bin/python
import argparse
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import jinja2
import rich
from loguru import logger
from omegaconf import OmegaConf
from medical_reasoning.run import make_info  # type: ignore

SEP = 80 * "-" + "\n\n"
FLOW_SEP = 80 * "." + "\n"
ANS_DEL = {"short": "\n\nA: ", "full": "\n\nAnswer: "}
Q_DEL = {"short": "\n\nQ: ", "full": "\n\nQuestion: "}

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


def load_data(
        data_dir: Path, filter_info: Optional[str]
) -> (List[str], List[Dict[str, Any]]):
    # placeholders for the data + parameters
    answers = defaultdict(dict)
    questions = dict()
    for exp in data_dir.iterdir():
        if not exp.is_dir():
            continue
        try:
            # data_file = exp / "data.json"
            config_file = exp / "config.yaml"
            cfg = OmegaConf.load(config_file)
            strategy = cfg["strategy"]["prompt"]
            strategy = strategy.replace("Let’s", "Let's")
            ans_del = ANS_DEL[cfg["prompt_style"]]
            q_del = Q_DEL[cfg["prompt_style"]]
            if args.filter_info is not None and args.filter_info != cfg.info:
                logger.info(f"Skipping ({cfg.info}): {exp}")
                continue
            for record in (exp / "output").iterdir():
                if not record.name.endswith(".txt"):
                    continue

                # read and parse the record
                content = open(record, "r").read()
                match = locator_re.search(content)
                dset, split, idx = match.groups()
                dset = formatted_dset[dset]
                content_parts = content.split(SEP)
                is_correct = content_parts[0].split(" ")[-1].replace("\n", "")
                is_correct = {"correct": True, "incorrect": False}[is_correct]
                ground_truth, answer_pred, *_ = content_parts[1].split("\n")
                ground_truth = ground_truth.replace("Answer: ", "")
                # answer_pred = answer_pred.replace("Prediction: ", "")

                flow = content.split(FLOW_SEP)[-1]
                try:
                    flow_parts = flow.split(ans_del)
                    answer = ans_del.replace("\n", "") + flow_parts[-1]
                    question = ans_del.join(flow_parts[:-1])
                    # split questions
                    question_parts = question.split(q_del)
                    if len(question_parts) > 1:
                        question = f"<small><i>{q_del.join(question_parts[:-1])}</small></i>{q_del}{question_parts[-1]}"
                except Exception as exc:
                    rich.print(len(flow.split(ans_del)))
                    logger.error(exc)

                # parse the ground truth and the pred
                outcome = "&#9989; " if is_correct else "&#10060; "

                # emphasis the right answer
                formatted_ground_truth = ground_truth.replace(': ', ') ')
                question = question.replace(
                    formatted_ground_truth, rf"<strong>{formatted_ground_truth}</strong>"
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
    strategies = list(
        sorted(
            strategies, key=lambda s: ORDERING.index(s) if s in ORDERING else math.inf
        )
    )
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
    output_path = data_dir / Path(args.fname)
    logger.info(f"Reading data from {data_dir}")

    # load the data
    strategies, data = load_data(data_dir, args.filter_info)

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
    parser.add_argument(
        "--filter_info", help="keep only run with info matching this", default=None
    )
    parser.add_argument("--fname", help="output file name", default="main.html")
    args = parser.parse_args()
    make_template(args)

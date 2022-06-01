__author__ = "Qiao Jin"

from datasets import Dataset

"""
Split the ori dataset to 500 test and 500 CV
Split the 500 CV to 10 folds
"""

from functools import reduce
import json
import math
import os
import random
import shutil
import sys


def split_pubmed(dataset: Dataset, fold):
    """
    dataset: dataset dict
    fold: number of splits
    output list of splited datasets
    Split the dataset for each label to ensure label proportion of different subsets are similar
    """
    random.seed(0)

    def add(x):
        return reduce(lambda a, b: a + b, x)

    label2pmid = {"yes": [], "no": [], "maybe": []}
    for pmid, label in enumerate(dataset["final_decision"]):
        label2pmid[label].append(pmid)

    label2pmid = {k: split_label(v, fold) for k, v in label2pmid.items()}  # splited

    output = []

    for i in range(fold):
        pmids = add([v[i] for _, v in label2pmid.items()])
        output.append(pmids)

    if len(output[-1]) != len(
        output[0]
    ):  # imbalanced: [51, 51, 51, 51, 51, 51, 51, 51, 51, 41]
        # randomly pick one from each to the last
        for i in range(fold - 1):
            pmids = list(output[i])
            picked = random.choice(pmids)
            output[-1][picked] = output[i][picked]
            output[i].pop(picked)

    return [dataset.select(pmids) for pmids in output]


def split_label(pmids, fold):
    """
    pmids: a list of pmids (of the same label)
    fold: number of splits
    output: list of split lists
    """
    random.shuffle(pmids)

    num_all = len(pmids)
    num_split = math.ceil(num_all / fold)

    output = []
    for i in range(fold):
        if i == fold - 1:
            output.append(pmids[i * num_split :])
        else:
            output.append(pmids[i * num_split : (i + 1) * num_split])

    return output

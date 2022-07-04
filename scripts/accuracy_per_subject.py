from pathlib import Path

import seaborn as sns
from datasets import load_dataset
from matplotlib import pyplot as plt

from .utils import get_base_arg_parser
from .utils import load_experiment_data

sns.set()

INFO2LABEL = {
    "full": "w/o context",
    "full-1docs": "w context",
}


class MedMcQaTopicLookup(object):
    def __init__(self, split="validation", **kwargs):
        self.split = split
        medmcqa = load_dataset("medmcqa", split=split, **kwargs)
        self.lookup = {}
        for i in range(len(medmcqa)):
            row = medmcqa[i]
            qid = row["id"]
            subject = row["subject_name"]
            if subject is None:
                subject = "--"
            self.lookup[qid] = subject

    def __call__(self, qid):
        qid = qid.replace(f"medmcqa_{self.split}-", "")
        return self.lookup[qid]


def run():
    arg_parser = get_base_arg_parser()
    args = arg_parser.parse_args()
    summary, records = load_experiment_data(args, compute_length=False)
    lookup = MedMcQaTopicLookup()
    records["subject"] = records["qid"].apply(lookup)

    records["hit"] = records.apply(lambda x: x["labels"] == x["predictions"], axis=1)
    records["info"] = records["info"].apply(lambda x: INFO2LABEL[x])
    records_per_bucket = records.groupby(
        [
            "subject",
            "info",
            "strategy",
        ]
    )["hit"].mean()
    records_per_bucket = records_per_bucket.reset_index()

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_args = {
        "x": "subject",
        "y": "hit",
        "hue": "info",
        "data": records_per_bucket,
        "ax": ax,
    }
    sns.boxplot(**plot_args)
    # sns.stripplot(**plot_args, size=4, color=".3", linewidth=0)

    ax.set_xlabel("Topic")
    ax.set_ylabel("Accuracy")
    # ax.set_title("Accuracy per Topic")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(args.path) / "accuracy-per-subject.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run()

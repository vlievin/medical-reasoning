from pathlib import Path

import numpy as np
import rich
from matplotlib import pyplot as plt

from .utils import load_experiment_data, get_base_arg_parser
import seaborn as sns
sns.set()

INFO2LABEL = {"full": "w/o context",
              "full-1docs": "w context",
              }

def run():
    arg_parser = get_base_arg_parser()
    args = arg_parser.parse_args()
    summary, records = load_experiment_data(args)
    print(summary)
    print(records)

    all_n_tokens = records["n_tokens"].values
    n_buckets = 4
    percentiles = [int(np.percentile(all_n_tokens, p))
                   for p in np.linspace(0, 100, n_buckets + 1)]
    rich.print(f"Percentiles: {percentiles}")
    def tok2bucket(n_tokens):
        for i in range(n_buckets):
            if n_tokens < percentiles[i]:
                return i
        return n_buckets - 1

    bucket2label = {i: f"{percentiles[i]:.0f} < x < {percentiles[i+1]:.0f}"
              for i in range(n_buckets)}

    records["n_tokens_bucket"] = records["n_tokens"].apply(tok2bucket)
    records["hit"] = records.apply(lambda x: x["labels"] == x["predictions"], axis=1)
    records["info"] = records["info"].apply(lambda x: INFO2LABEL[x])
    records_per_bucket = records.groupby(["n_tokens_bucket",
                                          "info",
                                          "strategy",])["hit"].mean()
    records_per_bucket = records_per_bucket.reset_index()

    g = sns.lineplot(x="n_tokens_bucket",
                 y="hit",
                 hue="info",
                # style="strategy",
                 data=records_per_bucket,
                 markers=True, dashes=False,
                 )
    g.set_xlabel("Number of tokens")
    g.set_ylabel("Accuracy")
    g.set_title("Accuracy vs. number of tokens")
    xlables = sorted(records_per_bucket["n_tokens_bucket"].unique())
    g.set_xticks(xlables)
    g.set_xticklabels([bucket2label[i] for i in xlables], rotation=45)
    plt.tight_layout()
    plt.savefig(Path(args.path) / "n_tokens_per_bucket.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run()
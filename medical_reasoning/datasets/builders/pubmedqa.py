"""PubMedQA: A Dataset for Biomedical Research Question Answering"""
import json
from pathlib import Path

import datasets
import rich
from datasets import Split

_CITATION = """\
@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing
  and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2567--2577},
  year={2019}
}
"""  # noqa

_DESCRIPTION = """\
We introduce PubMedQA, a novel biomedical question answering (QA) dataset collected from PubMed abstracts.
The task of PubMedQA is to answer research questions with yes/no/maybe
(e.g.: Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?)
using the corresponding abstracts. PubMedQA has 1k expert-annotated, 61.2k unlabeled
and 211.3k artificially generated QA instances.
Each PubMedQA instance is composed of (1) a question which is either an existing research article
title or derived from one, (2) a context which is the corresponding abstract without its conclusion,
(3) a long answer, which is the conclusion of the abstract and, presumably, answers the research question, and
(4) a yes/no/maybe answer which summarizes the conclusion.
PubMedQA is the first QA dataset where reasoning over biomedical research texts,
especially their quantitative contents, is required to answer the questions.
Our best performing model, multi-phase fine-tuning of BioBERT with long answer
bag-of-word statistics as additional supervision, achieves 68.1% accuracy,
compared to single human performance of 78.0% accuracy and majority-baseline
of 55.2% accuracy, leaving much room for improvement.
PubMedQA is publicly available at this https URL.
"""  # noqa

_HOMEPAGE = "https://pubmedqa.github.io/"

_LICENSE = """\

"""
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "pqa-l": "https://f001.backblazeb2.com/file/FindZebraData/fz-openqa/datasets/pubmedqa.zip"
}


class PubMedQAConfig(datasets.BuilderConfig):
    """BuilderConfig for MedQA"""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(PubMedQAConfig, self).__init__(
            version=datasets.Version("1.0.0", ""), **kwargs
        )


class PubMedQAConfig(datasets.GeneratorBasedBuilder):
    """ubMedQA: A Dataset for Biomedical Research Question Answering"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        PubMedQAConfig(
            name="pqa-l",
            description="PubMedQA-L : labelled set",
        ),
    ]
    ALLOWED_OPTIONS = ["yes", "no", "maybe"]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("int32"),
                    "uid": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer_idx": datasets.Value("int32"),
                    "options": datasets.Sequence(datasets.Value("string")),
                    "reasoning": datasets.Value("string"),
                    "documents": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        url = _URLS[self.config.name]
        downloaded_files_root = dl_manager.download_and_extract(url)
        downloaded_files = {
            split: Path(downloaded_files_root) / f"{split}_set.json"
            for split in [Split.TRAIN, Split.VALIDATION, Split.TEST]
        }
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepath": file, "split": split},
            )
            for split, file in downloaded_files.items()
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        data = json.load(open(filepath))
        for i, (uid, row) in enumerate(data.items()):
            answer = row["final_decision"]
            yield i, {
                "idx": i,
                "uid": f"{split}-{uid}",
                "question": row["QUESTION"],
                "documents": ["\n".join(row["CONTEXTS"])],
                "reasoning": row["LONG_ANSWER"],
                "options": self.ALLOWED_OPTIONS,
                "answer_idx": self.ALLOWED_OPTIONS.index(answer),
            }

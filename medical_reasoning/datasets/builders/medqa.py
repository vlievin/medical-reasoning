"""MedQA: What Disease does this Patient Have? A Large-scale Open Domain Question
Answering Dataset from Medical Exams"""
import json

import datasets

_CITATION = """\
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question
  Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang,
  Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
"""

_DESCRIPTION = """\
Open domain question answering (OpenQA) tasks have been recently attracting more and more attention
from the natural language processing (NLP) community. In this work, we present the first free-form
multiple-choice OpenQA dataset for solving medical problems, MedQA, collected from the professional
medical board exams. It covers three languages: English, simplified Chinese, and traditional
Chinese, and contains 12,723, 34,251, and 14,123 questions for the three languages, respectively.
We implement both rule-based and popular neural methods by sequentially combining a document
retriever and a machine comprehension model. Through experiments, we find that even the current
best method can only achieve 36.7%, 42.0%, and 70.1% of test accuracy on the English,
traditional Chinese, and simplified Chinese questions, respectively. We expect MedQA to present
great challenges to existing OpenQA systems and hope that it can serve as a platform to promote
much stronger OpenQA models from the NLP community in the future.
"""

_HOMEPAGE = "https://github.com/jind11/MedQA"

_LICENSE = """\

"""
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = {
    "us": {
        "train": "https://drive.google.com/file/d/1WtMXouYplMfJcIyMetaiyNHCftPZs8X1/"
        "view?usp=sharing",
        "validation": "https://drive.google.com/file/d/19t7vJfVt7RQ-stl5BMJkO-YoAicZ0tvs/"
        "view?usp=sharing",
        "test": "https://drive.google.com/file/d/1zxJOJ2RuMrvkQK6bCElgvy3ibkWOPfVY/"
        "view?usp=sharing",
    },
    "tw": {
        "train": "https://drive.google.com/file/d/1RPQJEu2iRY-KPwgQBB2bhFWY-LJ-z9_G/"
        "view?usp=sharing",
        "validation": "https://drive.google.com/file/d/1e-a6nE_HqnoQV_8k4YmaHbGSTTleM4Ag/"
        "view?usp=sharing",
        "test": "https://drive.google.com/file/d/13ISnB3mk4TXgqfu-JbsucyFjcAPnwwMG/"
        "view?usp=sharing",
    },
}


class MedQAConfig(datasets.BuilderConfig):
    """BuilderConfig for MedQA"""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(MedQAConfig, self).__init__(
            version=datasets.Version("1.0.0", ""), **kwargs
        )


class MedQA(datasets.GeneratorBasedBuilder):
    """MedQA: A Dataset for Biomedical Research Question Answering"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        MedQAConfig(
            name="us",
            description="USMLE MedQA dataset (English)",
        ),
        MedQAConfig(
            name="tw",
            description="TWMLE MedQA dataset (English - translated from Traditional Chinese)",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "question_clean": datasets.Value("string"),
                    "answer_idx": datasets.Value("int32"),
                    "options": datasets.Sequence(datasets.Value("string")),
                    "reasoning": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    @staticmethod
    def _get_drive_url(url):
        base_url = "https://drive.google.com/uc?id="
        split_url = url.split("/")
        return base_url + split_url[5]

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = {
            split: dl_manager.download_and_extract(self._get_drive_url(url))
            for split, url in _URLs[self.config.name].items()
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
        with open(filepath, "r") as f:
            for i, line in enumerate(f.readlines()):
                d = json.loads(line)
                # get raw data
                question = d["question"]
                answer = d["answer"]
                metamap = " ".join(d.get("metamap_phrases", []))
                options = list(d["options"].values())
                target = options.index(answer)

                assert len(options) == 4
                yield i, {
                    "idx": i,
                    "question": question,
                    "question_clean": metamap,
                    "answer_idx": target,
                    "options": options,
                    "reasoning": "",
                }

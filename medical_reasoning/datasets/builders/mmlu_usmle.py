import csv

import datasets

_CITATION = """\
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},

  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika
  and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and
  Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
"""

_DESCRIPTION = """\

"""

_HOMEPAGE = "https://github.com/hendrycks/test"

_LICENSE = """\

"""
_URLs = {
    datasets.Split.TEST: "https://drive.google.com/file/d/"
    "1rxo20nRbjCxYgB7uy1Fd1PEX4g0ALT-Z/view?usp=sharing",
    datasets.Split.TRAIN: "https://drive.google.com/file/d/"
    "1bpRqgqaZ6XkCvf2uUbQpzc6lYD21y-tm/view?usp=sharing",
    datasets.Split.VALIDATION: "https://drive.google.com/file/d/"
    "1sRYECC7-ZtZ_3rEKoxrBtikngnhsbi-m/view?usp=sharing",
}


class MMLU_USMLE(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

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
            for split, url in _URLs.items()
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
            for i, line in enumerate(csv.reader(f, delimiter=",")):
                question, *options, answer_str = line
                target = ["A", "B", "C", "D"].index(answer_str)

                assert len(options) == 4
                yield i, {
                    "idx": i,
                    "uid": f"{split}-{i}",
                    "question": question,
                    "answer_idx": target,
                    "options": options,
                    "reasoning": "",
                }

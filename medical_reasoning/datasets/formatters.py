import abc
from typing import Dict

import rich
from datasets import Dataset
from datasets import DatasetDict
from loguru import logger


class NestColumns(object):
    def __init__(self, input_columns, output_column):
        self.input_columns = input_columns
        self.output_column = output_column

    def __call__(self, row: Dict) -> Dict:
        output = [row[c] for c in self.input_columns]
        return {self.output_column: output}


class ConvertYesNoMaybe(object):
    def __init__(self, input_column):
        self.input_column = input_column

    def __call__(self, row: Dict) -> Dict:
        yesno_answer = row[self.input_column]
        options = ["no", "yes", "maybe"]
        answer = options.index(yesno_answer)
        return {"options": options, "answer": answer}


class FlattenPubmedqaContext(object):
    def __call__(self, row: Dict) -> Dict:
        documents = row["context"]["contexts"]
        return {"documents": documents}


class ConvertHeadQA(object):
    def __call__(self, row: Dict) -> Dict:
        r_index = row["ra"] - 1
        options = row["answers"]
        options = [x["atext"] for x in sorted(options, key=lambda x: x["aid"])]
        return {"ra": r_index, "answers": options}


class Formatter(object):
    @abc.abstractmethod
    def format(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError()

    def __call__(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        return DatasetDict(
            {split: self.format(dset) for split, dset in dataset.items()}
        )


class MedMCQAFormatter(Formatter):
    def format(self, dataset: Dataset, **kwargs) -> Dataset:
        # nest the answer options
        dataset = dataset.map(
            NestColumns(["opa", "opb", "opc", "opd"], "options"),
            desc="Nesting answer options",
            num_proc=4,
        )
        dataset = dataset.rename_columns(
            {
                "cop": "answer",
            }
        )
        dataset = dataset.remove_columns(["opa", "opb", "opc", "opd"])

        return dataset


class PubMedQAFormatter(Formatter):
    def format(self, dataset: Dataset, **kwargs) -> Dataset:
        # convert the yes/no answer
        dataset = dataset.map(
            ConvertYesNoMaybe("final_decision"),
            desc="Converting yes/no answers",
            num_proc=4,
        )

        dataset = dataset.map(
            FlattenPubmedqaContext(),
            desc="Flatten contexts",
            num_proc=4,
        )
        return dataset


class HeadQAFormatter(Formatter):
    def format(self, dataset: Dataset, **kwargs) -> Dataset:
        dataset = dataset.map(
            ConvertHeadQA(),
            desc="Formatting answers",
            num_proc=4,
        )

        dataset = dataset.rename_columns(
            {
                "qtext": "question",
                "ra": "answer",
                "answers": "options",
            }
        )
        dataset = dataset.remove_columns(["image"])
        return dataset

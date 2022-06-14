import abc
import re
from typing import Dict

from datasets import Dataset
from datasets import DatasetDict
from datasets import Split

from medical_reasoning.datasets.utils.split_pubmed import split_pubmed

# https://regex101.com/r/9YNTyr/1
medmcqa_ans_pattern = re.compile(
    (
        r"^((ans|answer)?(\.|:|-)?( *)?(is )?)?"
        r"((\(|\"| |')?[a-d](?!\w))(\)|\"| |')?"
        r"([ ]+i.e.[(,|.)])?( +)?"
    ),
    flags=re.IGNORECASE,
)


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
        options = ["yes", "no", "maybe"]
        answer = options.index(yesno_answer)
        return {"options": options, "answer_idx": answer}


class FlattenPubmedqaContext(object):
    def __call__(self, row: Dict) -> Dict:
        documents = row["context"]["contexts"]
        documents = ["\n".join(documents)]
        return {"documents": documents}


class AddIdx(object):
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def __call__(self, row: Dict, idx) -> Dict:
        return {self.column: f"{idx}-{self.value}"}


class ConvertHeadQA(object):
    def __call__(self, row: Dict) -> Dict:
        r_index = row["ra"] - 1
        options = row["answers"]
        options = [x["atext"] for x in sorted(options, key=lambda x: x["aid"])]
        if len(options) < 5:
            options += [""] * (5 - len(options))
        return {"ra": r_index, "answers": options}


class CleanuMedMCQAReasoning(object):
    def __init__(self, reasoning_column: str = "reasoning"):
        self.reasoning_column = reasoning_column

    def __call__(self, row: Dict) -> Dict:
        reasoning = row[self.reasoning_column]
        if reasoning is None:
            cleaned_reasoning = ""
        else:
            cleaned_reasoning = re.sub(medmcqa_ans_pattern, "", reasoning)
            # color = "red" if "ans" in cleaned_reasoning.lower() else "green"
            # rich.print(
            #     f"[gray]>>> {reasoning}\n"
            #     f"[{color}]>> {len(cleaned_reasoning)} "
            #     f">> ({type(cleaned_reasoning)}) {cleaned_reasoning}"
            # )
        return {self.reasoning_column: cleaned_reasoning}


class Formatter(object):
    @abc.abstractmethod
    def format(self, dataset: Dataset, split: Split, **kwargs) -> Dataset:
        raise NotImplementedError()

    def __call__(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        return DatasetDict(
            {
                split: self.format(dset, split=split, **kwargs)
                for split, dset in dataset.items()
            }
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
                "cop": "answer_idx",
                "exp": "reasoning",
                "id": "uid",
            }
        )
        dataset = dataset.remove_columns(["opa", "opb", "opc", "opd"])
        # cleanup reasoning
        dataset = dataset.map(
            CleanuMedMCQAReasoning(),
            desc="Cleaning up reasoning",
            num_proc=4,
        )
        return dataset


class PubMedQAFormatter(Formatter):
    def __call__(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        dataset = self.exctract_splits(dataset)
        return DatasetDict(
            {split: self.format(dset) for split, dset in dataset.items()}
        )

    @staticmethod
    def exctract_splits(dataset: DatasetDict) -> DatasetDict:
        assert set(dataset.keys()) == {Split.TRAIN}
        dataset = dataset[Split.TRAIN]

        train_dev, test = split_pubmed(dataset, 2)
        train, valid = split_pubmed(train_dev, 2)
        return DatasetDict(
            {
                Split.TRAIN: train,
                Split.VALIDATION: valid,
                Split.TEST: test,
            }
        )

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
        dataset = dataset.rename_columns(
            {
                "long_answer": "reasoning",
                "pubid": "uid",
            }
        )
        return dataset


class HeadQAFormatter(Formatter):
    def format(self, dataset: Dataset, split: Split, **kwargs) -> Dataset:
        dataset = dataset.map(
            ConvertHeadQA(),
            desc="Formatting answers",
            num_proc=4,
        )

        dataset = dataset.rename_columns(
            {
                "qtext": "question",
                "ra": "answer_idx",
                "answers": "options",
                "qid": "uid",
            }
        )
        dataset = dataset.map(
            AddIdx("uid", str(split)), desc="Concat uid", with_indices=True, num_proc=4
        )
        dataset = dataset.remove_columns(["image"])
        dataset = dataset.add_column("reasoning", [""] * len(dataset))
        return dataset

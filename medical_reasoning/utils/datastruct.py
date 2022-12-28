from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import rich
from loguru import logger
from pydantic import BaseModel
from pydantic import root_validator
from pydantic import validator


class Example(BaseModel):
    """A simple data structure for a single example."""

    question: str
    uid: str
    options: List[str]
    documents: List[str] = None
    reasoning: Optional[str] = None
    option_symbols: List[str]
    answer_idx: int
    question_clean: Optional[str] = None

    @property
    def answer_symbol(self):
        return self.option_symbols[self.answer_idx]

    @property
    def answer(self):
        return self.options[self.answer_idx]

    @validator("documents", always=True)
    def set_document_default_value(cls, value):
        if value is None:
            return []
        return value

    @root_validator()
    def ensure_answer_idx_range(cls, values: Dict[str, Optional[str]]):
        """Check if answer_idx is in range of options."""
        answer_idx = int(values.get("answer_idx"))
        option_symbols = values.get("option_symbols")
        if not isinstance(option_symbols, list):
            raise ValueError("`option_symbols` must be a list")
        if (answer_idx < 0 and answer_idx != -1) or answer_idx >= len(option_symbols):
            raise ValueError(
                f"Invalid answer_idx range: {answer_idx} not int [0, {len(option_symbols) - 1}]."
                f" Allowed options: {option_symbols}"
            )
        return values

    @root_validator()
    def ensure_number_of_options(cls, values: Dict[str, Optional[str]]):
        """Check if number of options is equal to number of allowed options."""
        options = values.get("options")
        option_symbols = values.get("option_symbols")
        if len(options) != len(option_symbols):
            raise ValueError(
                f"Invalid number of options: {len(options)} != {len(option_symbols)}. "
                f"Allowed options symbols: {option_symbols}, options: {options}"
            )
        return values


def permute_eg(eg: Example, seed: int = None) -> Example:
    answer = eg.options[eg.answer_idx]
    # get the shuffled indices
    rgn = np.random.RandomState(seed)
    ids = list(range(len(eg.options)))
    shuffled_ids = copy(ids)
    rgn.shuffle(shuffled_ids)
    id2shuffled_id = {shuffled_id: id for id, shuffled_id in zip(ids, shuffled_ids)}

    # make the new eg
    options = [eg.options[id] for id in shuffled_ids]
    if eg.documents is not None and len(eg.documents) == len(eg.options):
        # hack to keep documents aligned with options
        documents = [eg.documents[id] for id in shuffled_ids]
    else:
        documents = eg.documents
    answer_idx = id2shuffled_id[eg.answer_idx]

    new_eg = eg.copy(
        update={
            "options": options,
            "answer_idx": answer_idx,
            "documents": documents,
        }
    )
    new_answer = new_eg.options[new_eg.answer_idx]
    assert new_answer == answer, f"Answer mismatch: {new_answer} != {answer}"
    return new_eg


class Prediction(BaseModel):
    """A simple data structure for a single prediction."""

    prediction_str: Optional[str]
    example: Example
    meta: Optional[Dict[str, Any]]
    prediction_idx: int = -1
    label: str = "N/A"
    probs: Optional[List[float]] = None
    prediction_idx_per_sample: Optional[List[int]] = None

    @property
    def idx(self):
        return self.prediction_idx

    @property
    def repr(self):
        return self.prediction_str

    @property
    def full(self):
        return self.example.options[self.idx]

    @property
    def outcome(self):
        if self.idx == self.example.answer_idx:
            return "correct"
        return "incorrect"

    @root_validator()
    def parse_prediction_str(cls, values: Dict[str, Optional[str]]):
        """Check if answer_idx is in range of options."""
        eg = values["example"]
        prediction_str = values["prediction_str"]
        meta = values["meta"]

        # infer the prediction idx
        try:
            prediction_idx = eg.option_symbols.index(prediction_str)
            label = eg.option_symbols[prediction_idx]
        except Exception as exc:
            logger.warning(
                f"Prediction label couldn't be inferred "
                f"(prediction={prediction_str}, "
                f"option_symbols={eg.option_symbols}, "
                f"answer={meta.get('answer', 'N/A')}). "
                f"Exception: {exc}"
            )
            prediction_idx = -1
            label = "N/A"

        # clean the prediction str
        prediction_str = (
            eg.options[prediction_idx]
            if prediction_idx is not None and prediction_idx >= 0
            else "N/A"
        )

        # update the values
        values["prediction"] = prediction_str
        values["prediction_idx"] = prediction_idx
        values["label"] = label

        return values

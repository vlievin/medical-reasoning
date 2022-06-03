from typing import Dict
from typing import List
from typing import Optional

import rich
from loguru import logger
from pydantic import BaseModel
from pydantic import root_validator
from pydantic import validator


class Example(BaseModel):
    """A simple data structure for a single example."""

    question: str
    options: List[str]
    documents: List[str] = None
    allowed_options: List[str]
    answer_idx: int
    question_clean: Optional[str] = None

    @property
    def answer_symbol(self):
        return self.allowed_options[self.answer_idx]

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
        answer_idx = values.get("answer_idx")
        allowed_options = values.get("allowed_options")
        if answer_idx < 0 or answer_idx >= len(allowed_options):
            raise ValueError(
                f"Invalid answer_idx range: {answer_idx} not int [0, {len(allowed_options) - 1}]."
                f" Allowed options: {allowed_options}"
            )
        return values

    @root_validator()
    def ensure_number_of_options(cls, values: Dict[str, Optional[str]]):
        """Check if number of options is equal to number of allowed options."""
        options = values.get("options")
        allowed_options = values.get("allowed_options")
        if len(options) != len(allowed_options):
            raise ValueError(
                f"Invalid number of options: {len(options)} != {len(allowed_options)}. "
                f"Allowed options: {allowed_options}, options: {options}"
            )
        return values


class Prediction(BaseModel):
    """A simple data structure for a single prediction."""

    prediction_str: Optional[str]
    example: Example
    meta: Optional[Dict[str, str]]
    prediction_idx: int = -1
    label: str = "N/A"

    @property
    def idx(self):
        return self.prediction_idx

    @property
    def repr(self):
        return self.prediction_str

    @property
    def full(self):
        return self.prediction_str

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
            prediction_idx = eg.allowed_options.index(prediction_str)
            label = eg.allowed_options[prediction_idx]
        except Exception as exc:
            logger.warning(
                f"Prediction label couldn't be inferred "
                f"(prediction={prediction_str}, "
                f"allowed_options={eg.allowed_options}, "
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

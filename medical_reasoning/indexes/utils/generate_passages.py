from copy import copy
from typing import Dict, List, Any, Iterator


class GeneratePassages(object):
    def __init__(
        self,
        content_key: str = "text",
        passage_length: int = 100,
        passage_stride: int = 50,
    ):
        self.content_key = content_key
        self.passage_length = passage_length
        self.passage_stride = passage_stride

    def __call__(self, batch: Dict[str, List[Any]], **kwargs) -> Dict[str, List[Any]]:
        keys = list(batch.keys())
        batch_size = len(batch["text"])
        documents = [
            {key: batch[key][i] for key in batch.keys()} for i in range(batch_size)
        ]
        passages = [
            passage
            for document in documents
            for passage in self.yield_passages(document)
        ]

        return {key: [passage[key] for passage in passages] for key in keys}

    def yield_passages(self, document: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        content = document[self.content_key]
        content_words = content.split()
        content_length = len(content_words)
        for i in range(0, content_length - self.passage_length, self.passage_stride):
            passage = copy(document)
            passage.pop(self.content_key)
            passage_content = content_words[i : i + self.passage_length]
            if i > 0:
                passage_content = ["..."] + passage_content
            if i + self.passage_length < content_length:
                passage_content = passage_content + ["..."]
            passage[self.content_key] = " ".join(passage_content)
            yield passage
import json
from copy import copy
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional

import datasets
import rich
from elasticsearch import Elasticsearch
from loguru import logger

from medical_reasoning.indexes.utils.elasticsearch import es_create_index
from medical_reasoning.indexes.utils.elasticsearch import es_ingest_bulk
from medical_reasoning.indexes.utils.elasticsearch import es_remove_index
from medical_reasoning.indexes.utils.elasticsearch import es_search_bulk


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


class ElasticsearchIndex(object):
    _instance = None

    def __init__(
        self,
        *,
        dataset="wikipedia",
        name="20220301.en",
        subset: Optional[int] = None,
        ingest_bs: int = 1000,
        cache_dir: Optional[str] = None,
        num_proc: int = 4,
        title_boost_weight: float = 1.0,
        passage_length: int = 100,
        passage_stride: int = 50,
        es_body: Optional[Dict[str, Any]] = None,
    ):
        self.title_boost_weight = title_boost_weight
        # process the dataset
        wiki = datasets.load_dataset(dataset, name, cache_dir=cache_dir)
        wiki = datasets.concatenate_datasets(list(wiki.values()))
        if subset is not None:
            wiki = wiki.select(list(range(subset)))
            wiki._fingerprint = f"{wiki._fingerprint}-s{subset}"
        # generate passages
        pipe = GeneratePassages(
            passage_length=passage_length, passage_stride=passage_stride
        )
        wiki = wiki.map(
            pipe,
            batched=True,
            num_proc=num_proc,
            desc="Generating passages",
            batch_size=10,
        )
        # define the index name
        es_body_hash = hash(json.dumps(es_body))
        self.index_name = f"wikipedia-20220301.en-{wiki._fingerprint}-{es_body_hash}"
        # maybe create the index
        newly_created = es_create_index(self.instance, self.index_name, body=es_body)
        if newly_created:
            try:
                for i in range(0, len(wiki), ingest_bs):
                    batch = wiki[i : i + ingest_bs]
                    es_ingest_bulk(
                        self.instance,
                        self.index_name,
                        content=batch["text"],
                        title=batch["title"],
                    )
            except Exception as exc:
                es_remove_index(self.instance, self.index_name)
                raise exc
        else:
            logger.info(f"Index {self.index_name} already exists")

    @property
    def instance(self):
        if self._instance is None:
            self._instance = Elasticsearch()
        return self._instance

    def __call__(self, queries: List[str], query_titles: List[str], k: int = 10):
        return es_search_bulk(
            self.instance,
            index_name=self.index_name,
            queries=queries,
            title_queries=query_titles,
            title_boost=self.title_boost_weight,
            k=k,
        )

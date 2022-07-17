from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from datasets import Dataset
from elasticsearch import Elasticsearch
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from medical_reasoning.indexes.base import Index
from medical_reasoning.indexes.base import SearchResults
from medical_reasoning.indexes.utils.elasticsearch import es_create_index
from medical_reasoning.indexes.utils.elasticsearch import es_ingest_bulk
from medical_reasoning.indexes.utils.elasticsearch import es_remove_index
from medical_reasoning.indexes.utils.elasticsearch import es_search_bulk


def keep_only_alpha(s):
    return "".join(filter(str.isalpha, s))


class ElasticsearchIndex(Index):
    _instance = None

    def __init__(
        self,
        *,
        es_body: Optional[Dict[str, Any]] = None,
        column_name: str = "text",
        aux_weights: Optional[Dict[str, float]] = None,
        filter_numbers: bool = False,
        corpus_name: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._instance = None
        self.column_name = column_name
        self.filter_numbers = filter_numbers
        self.aux_weights = aux_weights

        # define the index name
        dset_name = corpus_name
        fingerprint = self.corpus._fingerprint
        index_name = f"{dset_name}_{fingerprint}"
        self.index_name = index_name.lower()

        # potentially create the index
        self.maybe_create_index(self.corpus, es_body)

    def maybe_create_index(
        self, corpus: Dataset, es_body: Dict[str, Any], ingest_bs: int = 1000
    ):
        if isinstance(es_body, DictConfig):
            es_body = OmegaConf.to_container(es_body)
        # maybe create the index
        newly_created = es_create_index(self.instance, self.index_name, body=es_body)
        if newly_created:
            try:
                for i in tqdm(
                    range(0, len(corpus), ingest_bs),
                    desc=f"Indexing {self.index_name} with Elasticsearch",
                ):
                    batch = corpus[i : i + ingest_bs]
                    es_ingest_bulk(
                        self.instance,
                        self.index_name,
                        content=batch["text"],
                        title=batch["title"],
                        idx=batch["id"],
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

    def __call__(
        self, queries: List[str], aux_queries: Optional[List[str]], *, k: int = 10
    ) -> SearchResults:

        if self.filter_numbers:
            queries = [self.filter_if_only_numbers(query) for query in queries]
            if aux_queries is not None:
                aux_queries = [
                    self.filter_if_only_numbers(query) for query in aux_queries
                ]

        # retrieve the top k nearest examples
        output = es_search_bulk(
            self.instance,
            index_name=self.index_name,
            queries=queries,
            aux_queries=aux_queries,
            aux_weights=self.aux_weights,
            k=k,
        )

        return SearchResults(**output)

    def filter_if_only_numbers(self, query):
        only_alpha = keep_only_alpha(query)
        if len(only_alpha) == 0:
            return ""
        else:
            return query

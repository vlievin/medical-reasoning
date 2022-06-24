from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import elasticsearch
from datasets import DatasetDict
from omegaconf import DictConfig
from omegaconf import OmegaConf

from medical_reasoning.indexes.base import Index
from medical_reasoning.indexes.base import SearchResults


def keep_only_alpha(s):
    return "".join(filter(str.isalpha, s))


class ElasticsearchIndex(Index):
    _instance = None

    def __init__(
        self,
        *,
        corpus: DictConfig | DatasetDict,
        subset: Optional[int] = None,
        prepare_corpus: bool = True,
        es_body: Optional[Dict[str, Any]] = None,
        column_name: str = "text",
        use_aux_queries: bool = False,
        filter_numbers: bool = False,
    ):
        super(ElasticsearchIndex, self).__init__(
            corpus=corpus, subset=subset, prepare_corpus=prepare_corpus
        )
        self.use_aux_queries = use_aux_queries
        self.column_name = column_name
        self.filter_numbers = filter_numbers
        if isinstance(es_body, DictConfig):
            es_body = OmegaConf.to_container(es_body)

        # add elasticsearch index
        dset_name = self.corpus.info.builder_name
        fingerprint = self.corpus._fingerprint
        index_name = f"{dset_name}_{fingerprint}"
        if self.column_name != "text":
            index_name += f"_{self.column_name}"

        try:
            self.corpus.load_elasticsearch_index(
                self.column_name,
                host="localhost",
                es_index_name=index_name,
                es_index_config=es_body,
            )
            # try searching something to trigger `NotFoundError`.
            self.corpus.get_nearest_examples(self.column_name, "this is a test.", k=1)
        except elasticsearch.exceptions.NotFoundError:
            self.corpus.add_elasticsearch_index(
                self.column_name,
                host="localhost",
                es_index_name=index_name,
                es_index_config=es_body,
            )

    def __call__(
        self, queries: List[str], aux_queries: Optional[List[str]], *, k: int = 10
    ) -> SearchResults:
        if self.use_aux_queries:
            queries = aux_queries

        if self.filter_numbers:
            queries = [self.cleanup_number(q) for q in queries]

        # retrieve the top k nearest examples
        batch_results = self.corpus.get_nearest_examples_batch(
            self.column_name, queries, k=k
        )

        return self._format_batch_results(batch_results)

    def cleanup_number(self, query):
        only_alpha = keep_only_alpha(query)
        if len(only_alpha) == 0:
            return ""
        else:
            return query

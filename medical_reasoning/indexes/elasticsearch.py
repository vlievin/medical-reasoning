from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from datasets import DatasetDict
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf

from medical_reasoning.indexes.base import Index
from medical_reasoning.indexes.base import SearchResults


class ElasticsearchIndex(Index):
    _instance = None

    def __init__(
        self,
        *,
        corpus: DictConfig | DatasetDict,
        subset: Optional[int] = None,
        prepare_corpus: bool = True,
        es_body: Optional[Dict[str, Any]] = None,
    ):
        super(ElasticsearchIndex, self).__init__(
            corpus=corpus, subset=subset, prepare_corpus=prepare_corpus
        )
        if isinstance(es_body, DictConfig):
            es_body = OmegaConf.to_container(es_body)

        # add elasticsearch index
        dset_name = self.corpus.info.builder_name
        fingerprint = self.corpus._fingerprint
        index_name = f"{dset_name}_{fingerprint}"
        try:
            self.corpus.load_elasticsearch_index(
                "text",
                host="localhost",
                es_index_name=index_name,
                es_index_config=es_body,
            )
        except Exception as exc:
            logger.warning(f"Failed to load elasticsearch index: {exc}")
            logger.info(f"Trying to create index {index_name}")
            self.corpus.add_elasticsearch_index(
                "text",
                host="localhost",
                es_index_name=index_name,
                es_index_config=es_body,
            )

    def __call__(self, queries: List[str], k: int = 10) -> SearchResults:

        # retrieve the top k nearest examples
        batch_results = self.corpus.get_nearest_examples_batch("text", queries, k=k)

        return self._format_batch_results(batch_results)

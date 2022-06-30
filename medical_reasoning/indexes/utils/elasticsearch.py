from typing import Dict
from typing import List
from typing import Optional

import rich
from elasticsearch import Elasticsearch
from elasticsearch import RequestError
from elasticsearch import helpers as es_helpers
from loguru import logger


def es_search_bulk(
        es_instance: Elasticsearch,
        *,
        index_name: str,
        queries: List[str],
        aux_queries: List[str],
        aux_weights: Optional[Dict[str, float]] = None,
        k: int = 10,
):
    """Batch query an ElasticSearch Index"""
    es_request = []
    for i, query in enumerate(queries):
        query_parts = []

        # this is the main query
        query_parts.append(
            {
                "match": {
                    "text": {
                        "query": query,
                        "operator": "or",
                    },
                }
            },
        )

        # these are the auxiliary queries
        if aux_queries is not None:
            if aux_weights is None:
                raise ValueError("aux_weights must be provided if aux_queries is not None")

            for field, weight in aux_weights.items():
                query_parts.append({
                    'match': {
                        field: {
                            "query": aux_queries[i],
                            "operator": "or",
                            "boost": weight,
                        }
                    }
                })

        # make the final request
        es_request_i = {
            "query": {
                "bool": {"should": query_parts},
            },
            "from": 0,
            "size": k,
        }

        # append the header and body of the request
        es_request.extend([{"index": index_name}, es_request_i])

    result = es_instance.msearch(body=es_request, index=index_name, request_timeout=600)

    indices, titles, scores, contents = [], [], [], []
    for query in result["responses"]:
        temp_indices, temp_titles, temp_scores, temp_content = [], [], [], []
        if "hits" not in query:
            rich.print("[magenta]===== ES RESPONSE =====")
            rich.print(query)
            rich.print("[magenta]=======================")
            raise ValueError("ES did not return any hits (see above for details)")

        for hit in query["hits"]["hits"]:
            temp_scores.append(hit["_score"])
            temp_indices.append(hit["_source"]["id"])
            temp_titles.append(hit["_source"]["title"])
            temp_content.append(hit["_source"]["text"])

        indices.append(temp_indices)
        titles.append(temp_titles)
        scores.append(temp_scores)
        contents.append(temp_content)

    return {
        "indices": indices,
        "titles": titles,
        "scores": scores,
        "texts": contents,
    }


def es_create_index(
        es_instance: Elasticsearch, index_name: str, body: Optional[Dict] = None
) -> bool:
    """
    Create ElasticSearch Index
    """
    try:
        response = es_instance.indices.create(index=index_name, body=body)
        logger.info(response)
        newly_created = True

    except RequestError as err:
        if err.error == "resource_already_exists_exception":
            newly_created = False
        else:
            raise err

    return newly_created


def es_remove_index(es_instance: Elasticsearch, index_name: str):
    """
    Remove ElasticSearch Index
    """
    return es_instance.indices.delete(index=index_name)


def es_ingest_bulk(
        es_instance: Elasticsearch,
        index_name: str,
        *,
        content: List[str],
        title: List[str],
        idx: List[str],
        chunk_size=1000,
        request_timeout=200,
):
    actions = [
        {
            "_index": index_name,
            "_title": title,
            "_source": {
                "title": title[i],
                "text": content[i],
                "id": idx[i],
            },
        }
        for i in range(len(content))
    ]

    return es_helpers.bulk(
        es_instance,
        actions,
        chunk_size=chunk_size,
        request_timeout=request_timeout,
        refresh="true",
    )

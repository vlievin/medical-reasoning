from typing import Dict
from typing import List
from typing import Optional

import rich
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from elasticsearch import RequestError
from loguru import logger


def es_search_bulk(
    es_instance: Elasticsearch,
    *,
    index_name: str,
    queries: List[str],
    title_queries: List[str],
    title_boost: float = 1.0,
    k: int = 10,
):
    """Batch query an ElasticSearch Index"""
    request = []
    for i, query in enumerate(queries):
        should_query_part = []
        if title_queries is not None:
            query = query + " " + title_queries[i]

        # this is the main query
        should_query_part.append(
            {
                "match": {
                    "text": {
                        "query": query,
                        # "zero_terms_query": "all",
                        "operator": "or",
                    },
                }
            },
        )

        if title_queries is not None:
            should_query_part.append(
                {
                    "match": {
                        "title": {
                            "query": title_queries[i],
                            # "zero_terms_query": "all",
                            "operator": "or",
                            "boost": title_boost,
                        },
                    }
                },
            )

        # final request
        r = {
            "query": {
                "bool": {"should": should_query_part},
            },
            "from": 0,
            "size": k,
        }

        # append the header and body of the request
        request.extend([{"index": index_name}, r])

    result = es_instance.msearch(body=request, index=index_name, request_timeout=200)

    titles, scores, contents = [], [], []
    for query in result["responses"]:
        temp_titles, temp_scores, temp_content = [], [], []
        if "hits" not in query:
            rich.print("[magenta]===== ES RESPONSE =====")
            rich.print(query)
            rich.print("[magenta]=======================")
            raise ValueError("ES did not return any hits (see above for details)")

        for hit in query["hits"]["hits"]:
            temp_scores.append(hit["_score"])
            temp_titles.append(hit["_source"]["title"])
            temp_content.append(hit["_source"]["text"])

        titles.append(temp_titles)
        scores.append(temp_scores)
        contents.append(temp_content)

    return {
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

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, cast
from elasticsearch import Elasticsearch
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class ElasticsearchRetriever(BaseRetriever):
    client: Elasticsearch = Field(..., description="Elasticsearch client instance.")
    index_name: Union[str, Sequence[str]] = Field(..., description="Name of the index or list of indices to query.")
    body_func: Callable[[str], Dict] = Field(..., description="Function that takes a query string and returns the Elasticsearch query body.")
    content_field: Optional[Union[str, Mapping[str, str]]] = Field(
        None,
        description="The field to use as the content of the documents.",
    )
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        index_name: Union[str, Sequence[str]],
        body_func: Callable[[str], Dict],
        *,
        content_field: Optional[Union[str, Mapping[str, str]]] = None,
        document_mapper: Optional[Callable[[Mapping], Document]] = None,
        client: Optional[Elasticsearch] = None,
    ) -> None:
        # Create client from credentials if needed (BEFORE super().__init__)
        es_connection = client

        super().__init__(
            client=es_connection,
            index_name=index_name,
            body_func=body_func,
            content_field=content_field,
            document_mapper=document_mapper,
        )

        # Now Pydantic has set everything, do validation
        if self.content_field is None:
            raise ValueError("content_field must be defined.")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        body = self.body_func(query)
        results = self.client.search(index=self.index_name, body=body)
        documents = []
        for hit in results["hits"]["hits"]:
            content = hit["_source"].pop(self.content_field)
            documents.append(Document(
                page_content=content,
                metadata=hit["_source"],
            ))
        return documents

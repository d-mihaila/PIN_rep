from typing import Any, Coroutine, List

import httpx
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from pydantic import Field, PrivateAttr, model_validator, BaseModel
from langchain_openai import ChatOpenAI   # modern LLM client
from .settings import EmmRetrieversSettings
from openai import OpenAI

# -----------------------------------------------------------------------
# Utility for converting dicts to LCEL Documents
# -----------------------------------------------------------------------

def as_lc_docs(dicts: list[dict]) -> list[Document]:
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"]) for d in dicts
    ]


class EmmRetrieverV1(BaseRetriever):
    settings: EmmRetrieversSettings
    spec: dict
    filter: dict | None = None
    params: dict = Field(default_factory=dict)
    route: str = "/r/rag-minimal/query"
    add_ref_key: bool = True

    _client: httpx.Client = PrivateAttr()
    _aclient: httpx.AsyncClient = PrivateAttr()

    # ------- interface impl:
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        r = self._client.post(**self.search_post_kwargs(query))
        if r.status_code == 422:
            print("ERROR:\n", r.json())
        r.raise_for_status()
        resp = r.json()
        return self._as_lc_docs(resp["documents"])

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> Coroutine[Any, Any, list[Document]]:
        r = await self._aclient.post(**self.search_post_kwargs(query))
        if r.status_code == 422:
            print("ERROR:\n", r.json())
        r.raise_for_status()
        resp = r.json()
        return self._as_lc_docs(resp["documents"])

    # ---------
    @model_validator(mode="after")
    def create_clients(self):
        _auth_headers = {
            "Authorization": f"Bearer {self.settings.API_KEY.get_secret_value()}"
        }

        kwargs = dict(
            base_url=self.settings.API_BASE,
            headers=_auth_headers,
            timeout=self.settings.DEFAULT_TIMEOUT,
        )

        self._client = httpx.Client(**kwargs)
        self._aclient = httpx.AsyncClient(**kwargs)
        return self

    @model_validator(mode="after")
    def apply_default_params(self):
        self.params = {
            **{
                "cluster_name": self.settings.DEFAULT_CLUSTER,
                "index": self.settings.DEFAULT_INDEX,
            },
            **(self.params or {}),
        }
        return self

    def _as_lc_docs(self, dicts: list[dict]) -> list[Document]:
        docs = as_lc_docs(dicts)
        if self.add_ref_key:
            for i, d in enumerate(docs):
                d.metadata["ref_key"] = i

        return docs

    def search_post_kwargs(self, query: str):
        return dict(
            url=self.route,
            params=self.params,
            json={"query": query, "spec": self.spec, "filter": self.filter},
        )


    
# Multi-query generator

class QueryList(BaseModel):
    queries: List[str] = Field(description="List of rewritten search queries")


def parse_multiquery_output(text: str) -> QueryList:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    print('to fetch relevant docs, we use the following questions: \n', lines)
    return QueryList(queries=lines)


QUERY_PROMPT = PromptTemplate.from_template("""
Generate 5 alternative search queries in different wording and angles.
They must refer to the same disaster event.
Write each query on a new line.

Original: {question}
""")


class EmmAugmentedRetriever(BaseRetriever):
    base_retriever: EmmRetrieverV1
    llm_chain: Any = None
    @model_validator(mode="after")
    def init_models(self):
        """Build the multiquery LLM chain"""

        api_key = self.base_retriever.settings.OPENAI_API_KEY
        base_url = self.base_retriever.settings.OPENAI_API_BASE_URL

        # 1. Multi-query rewriting LLM
        if self.llm_chain is None:
            self.llm_chain = (
                QUERY_PROMPT
                | ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    api_key=api_key,
                    base_url=base_url,
                )
                | StrOutputParser()
            )

        return self

    def _get_relevant_documents(self, query: str, run_manager=None):

        # ----- 1. Multi-query rewrite -----
        raw_output = self.llm_chain.invoke({"question": query})
        queries = parse_multiquery_output(raw_output).queries

        # ----- 2. Fetch documents for each rewritten query -----
        all_docs = []
        for q in queries:
            docs = self.base_retriever.invoke(q)
            all_docs.extend(docs)

        # ----- 3. Deduplicate -----
        seen = set()
        unique_docs = []
        for d in all_docs:
            key = (
                d.metadata.get("doc_id")
                or d.metadata.get("ref_key")
                or hash(d.page_content)
            )
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        print('relevant, non-duplicated documents found:', len(unique_docs))
        return unique_docs

    def invoke_json(self, query: str):
        docs = self.invoke(query)
        return {
            "documents": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in docs
            ]
        }

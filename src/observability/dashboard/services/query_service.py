"""Read-only dashboard queries use the same configurable retrieval core as MCP."""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter

from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import create_core_reranker
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.settings import Settings, resolve_path
from src.core.trace import TraceCollector, TraceContext
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


def seed_demo(settings: Settings) -> int:
    """Idempotently seed only the explicitly selected, isolated offline demo collection."""
    if (
        settings.embedding.provider != "local_hash"
        or settings.vector_store.collection_name != "publication_demo"
    ):
        raise ValueError("Demo loading is limited to the offline publication_demo profile")
    documents = json.loads(
        resolve_path("fixtures/publication_demo.json").read_text(encoding="utf-8")
    )
    embedding = EmbeddingFactory.create(settings)
    vectors = embedding.embed([doc["text"] for doc in documents])
    store = VectorStoreFactory.create(settings, collection_name="publication_demo")
    try:
        store.upsert(
            [
                {
                    "id": doc["id"],
                    "vector": vector,
                    "metadata": {
                        "text": doc["text"],
                        "title": doc["title"],
                        "source_path": doc["source"],
                        "fixture": "publication_demo_v1",
                    },
                }
                for doc, vector in zip(documents, vectors, strict=True)
            ]
        )
    finally:
        store.close()
    return len(documents)


def search(settings: Settings, query: str, collection: str, top_k: int = 5) -> dict:
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("Query must contain 1 to 2000 characters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,62}", collection):
        raise ValueError("Invalid collection name")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer between 1 and 10")
    start = perf_counter()
    trace = TraceContext(trace_type="query")
    trace.metadata.update(
        {"source": "dashboard", "retrieval_strategy": settings.retrieval.strategy}
    )
    store = VectorStoreFactory.create(settings, collection_name=collection)
    try:
        embedding = EmbeddingFactory.create(settings)
        dense = create_dense_retriever(
            settings=settings, embedding_client=embedding, vector_store=store
        )
        sparse = create_sparse_retriever(
            settings=settings,
            bm25_indexer=BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}"))),
            vector_store=store,
        )
        sparse.default_collection = collection
        engine = create_hybrid_search(
            settings=settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
        )
        result = engine.search(query=query.strip(), top_k=top_k, trace=trace, return_details=True)
        if result.dense_error and settings.retrieval.strategy == "dense":
            raise RuntimeError("Configured embedding/retrieval service is unavailable")
        rows = result.results
        rerank_fallback = False
        reranker = create_core_reranker(settings=settings)
        if settings.rerank.enabled and rows:
            reranked = reranker.rerank(query=query.strip(), results=rows, top_k=top_k, trace=trace)
            rows, rerank_fallback = reranked.results, reranked.used_fallback
        trace.finish()
        TraceCollector().collect(trace)
        return {
            "query": query.strip(),
            "collection": collection,
            "strategy": settings.retrieval.strategy,
            "provider": settings.embedding.provider,
            "elapsed_ms": round((perf_counter() - start) * 1000, 2),
            "used_fallback": result.used_fallback or rerank_fallback,
            "stages": [
                {"stage": s.get("stage"), "elapsed_ms": s.get("elapsed_ms")} for s in trace.stages
            ],
            "results": [
                {
                    "chunk_id": row.chunk_id,
                    "score": row.score,
                    "text": row.text,
                    "title": str(row.metadata.get("title", "来源片段")),
                    "source": Path(
                        str(row.metadata.get("source_path", "未标注来源")).replace("\\", "/")
                    ).name,
                }
                for row in rows[:top_k]
            ],
        }
    finally:
        store.close()

"""Resume-alignment contracts for the benchmark-selected retrieval strategy."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.trace import TraceContext
from src.core.types import ProcessedQuery, RetrievalResult
from src.libs.embedding.local_hash_embedding import LocalHashEmbedding


def _settings(strategy: str = "dense") -> SimpleNamespace:
    return SimpleNamespace(
        embedding=SimpleNamespace(provider="local_hash", dimensions=16),
        retrieval=SimpleNamespace(
            strategy=strategy,
            dense_top_k=20,
            sparse_top_k=20,
            fusion_top_k=10,
            rrf_k=60,
        ),
    )


def test_dense_strategy_skips_sparse_and_rrf() -> None:
    expected = RetrievalResult(chunk_id="dense-1", score=0.9, text="answer", metadata={})
    query_processor = MagicMock()
    query_processor.process.return_value = ProcessedQuery(
        original_query="answer",
        keywords=["answer"],
        filters={},
    )
    dense = MagicMock()
    dense.retrieve.return_value = [expected]
    sparse = MagicMock()
    fusion = MagicMock()
    trace = TraceContext(trace_type="query")
    search = HybridSearch(
        settings=_settings("dense"),
        query_processor=query_processor,
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=fusion,
    )

    assert search.search("answer", trace=trace) == [expected]
    dense.retrieve.assert_called_once()
    sparse.retrieve.assert_not_called()
    fusion.fuse.assert_not_called()
    assert trace.metadata["retrieval_strategy"] == "dense"


def test_hybrid_strategy_keeps_sparse_and_rrf_enabled() -> None:
    search = HybridSearch(settings=_settings("hybrid_rrf"))
    assert search.config.enable_dense is True
    assert search.config.enable_sparse is True
    assert search.config.parallel_retrieval is True


def test_local_hash_embedding_is_deterministic_and_normalised() -> None:
    embedding = LocalHashEmbedding(_settings())
    first, second = embedding.embed(["same technical query", "same technical query"])

    assert first == second
    assert len(first) == 16
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9

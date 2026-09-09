"""Runtime contracts for traces, public MCP schemas, and resource cleanup."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.types import RetrievalResult
from src.mcp_server.protocol_handler import ProtocolHandler, _register_default_tools
from src.mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubConfig,
    QueryKnowledgeHubTool,
)


class _DenseSearch:
    config = SimpleNamespace(strategy="dense")

    def search(self, *, query, top_k, filters, trace, return_details):
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.91,
                text="Grounded technical answer",
                metadata={"source_path": "docs/answer.md"},
            ),
            RetrievalResult(
                chunk_id="chunk-2",
                score=0.82,
                text="Supporting context",
                metadata={"source_path": "docs/context.md"},
            ),
        ]
        trace.record_stage(
            "dense_retrieval",
            {
                "method": "dense",
                "provider": "local_hash",
                "chunks": [
                    {"chunk_id": item.chunk_id, "score": item.score}
                    for item in results
                ],
            },
            elapsed_ms=0.25,
        )
        return results[:top_k]


@pytest.mark.asyncio
async def test_successful_query_trace_records_strategy_candidates_and_final_ranks() -> None:
    settings = SimpleNamespace(retrieval=SimpleNamespace(strategy="dense"))
    tool = QueryKnowledgeHubTool(
        settings=settings,
        config=QueryKnowledgeHubConfig(enable_rerank=False),
        hybrid_search=_DenseSearch(),
        reranker=MagicMock(),
    )
    tool._initialized = True
    tool._current_collection = "default"
    collector = MagicMock()

    with patch(
        "src.mcp_server.tools.query_knowledge_hub.TraceCollector",
        return_value=collector,
    ):
        response = await tool.execute("technical question", top_k=2)

    assert response.is_empty is False
    trace = collector.collect.call_args.args[0]
    assert trace.metadata["retrieval_strategy"] == "dense"
    assert [item["rank"] for item in trace.metadata["final_results"]] == [1, 2]
    stage = next(item for item in trace.stages if item["stage"] == "dense_retrieval")
    assert stage["elapsed_ms"] > 0
    assert stage["data"]["provider"] == "local_hash"
    assert [item["chunk_id"] for item in stage["data"]["chunks"]] == [
        "chunk-1",
        "chunk-2",
    ]


@pytest.mark.asyncio
async def test_public_query_schema_rejects_out_of_range_top_k_before_execution() -> None:
    handler = ProtocolHandler("test", "1")
    _register_default_tools(handler)
    try:
        result = await handler.execute_tool(
            "query_knowledge_hub",
            {"query": "question", "top_k": 21},
        )
    finally:
        handler.close()

    assert result.isError is True
    assert "must be <= 20" in result.content[0].text

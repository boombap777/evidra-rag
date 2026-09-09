"""Unit tests for the reproducible Stack Overflow benchmark helpers."""

from scripts.benchmark_stackoverflow import (
    aggregate_metrics,
    ranking_metrics,
    select_quality_gated_method,
    stable_id,
)


def test_stable_id_is_deterministic() -> None:
    assert stable_id("so", "same text") == stable_id("so", "same text")
    assert stable_id("so", "same text") != stable_id("so", "different text")


def test_ranking_metrics_with_first_rank_hit() -> None:
    metrics = ranking_metrics(["gold", "other"], ["gold"], k=2)
    assert metrics == {"hit_rate": 1.0, "recall": 1.0, "mrr": 1.0, "ndcg": 1.0}


def test_ranking_metrics_with_no_hit() -> None:
    metrics = ranking_metrics(["a", "b"], ["gold"], k=2)
    assert metrics == {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}


def test_aggregate_metrics_uses_mean() -> None:
    metrics = aggregate_metrics(
        [
            {"hit_rate": 1.0, "recall": 1.0, "mrr": 1.0, "ndcg": 1.0},
            {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
        ],
        k=10,
    )
    assert metrics["hit_rate_at_10"] == 0.5
    assert metrics["mrr_at_10"] == 0.5


def test_quality_gate_rejects_regressing_reranker() -> None:
    metrics = {
        "bm25": {"hit_rate_at_10": 0.70, "mrr_at_10": 0.40, "ndcg_at_10": 0.45},
        "dense": {"hit_rate_at_10": 0.84, "mrr_at_10": 0.53, "ndcg_at_10": 0.59},
        "hybrid_rrf": {"hit_rate_at_10": 0.83, "mrr_at_10": 0.48, "ndcg_at_10": 0.55},
        "hybrid_rrf_rerank": {
            "hit_rate_at_10": 0.81,
            "mrr_at_10": 0.49,
            "ndcg_at_10": 0.55,
        },
    }
    gate = select_quality_gated_method(metrics, k=10)
    assert gate["selected_method"] == "dense"
    assert gate["reranker_accepted"] is False
    assert gate["fallback_method"] == "dense"

#!/usr/bin/env python
"""Reproducible retrieval benchmark on Stack Overflow duplicate questions.

The benchmark builds one global corpus from a deterministic sample of the
public MTEB StackOverflowDupQuestions test split.  Each query is evaluated
against the full corpus (not only its original per-query candidate list) with
four retrieval strategies:

1. BM25
2. Qwen3 dense embeddings
3. BM25 + Dense + RRF
4. BM25 + Dense + RRF + Cross-Encoder

It writes the generated corpus, a labelled golden set, machine-readable
metrics, and a Markdown report under ``data/benchmarks/stackoverflow``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from src.core.query_engine.fusion import RRFFusion  # noqa: E402
from src.core.query_engine.query_processor import QueryProcessor  # noqa: E402
from src.core.settings import RerankSettings, load_settings  # noqa: E402
from src.core.types import Chunk, RetrievalResult  # noqa: E402
from src.ingestion.embedding.sparse_encoder import SparseEncoder  # noqa: E402
from src.ingestion.storage.bm25_indexer import BM25Indexer  # noqa: E402
from src.libs.embedding.embedding_factory import EmbeddingFactory  # noqa: E402
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402

DATASET_NAME = "mteb/stackoverflowdupquestions-reranking"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "benchmarks" / "stackoverflow"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def configure_console_encoding() -> None:
    """Use UTF-8 for direct Windows CLI runs without replacing pytest streams."""

    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--retrieval-depth", type=int, default=50)
    parser.add_argument("--metric-k", type=int, default=10)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--latency-queries", type=int, default=100)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def normalise_text(value: str) -> str:
    return " ".join(str(value).split())


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def load_public_benchmark(
    query_count: int,
    seed: int,
) -> tuple[dict[str, str], list[dict[str, Any]], int]:
    """Load a deterministic test sample and merge candidates into one corpus."""

    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split="test")
    if query_count < 1 or query_count > len(dataset):
        raise ValueError(f"queries must be in [1, {len(dataset)}], got {query_count}")

    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(dataset)), query_count)

    corpus: dict[str, str] = {}
    golden: list[dict[str, Any]] = []

    for ordinal, dataset_index in enumerate(selected_indices):
        row = dataset[dataset_index]
        query = normalise_text(row["query"])
        positives = [normalise_text(item) for item in row["positive"] if normalise_text(item)]
        negatives = [normalise_text(item) for item in row["negative"] if normalise_text(item)]

        if not query or not positives:
            continue

        positive_ids: list[str] = []
        candidate_ids: list[str] = []
        for text in positives + negatives:
            doc_id = stable_id("so", text)
            corpus[doc_id] = text
            candidate_ids.append(doc_id)
            if text in positives:
                positive_ids.append(doc_id)

        golden.append(
            {
                "query_id": f"q_{ordinal:04d}",
                "dataset_index": dataset_index,
                "query": query,
                "expected_chunk_ids": sorted(set(positive_ids)),
                "source_candidate_ids": sorted(set(candidate_ids)),
            }
        )

    return corpus, golden, len(dataset)


def write_dataset_artifacts(
    output_dir: Path,
    corpus: dict[str, str],
    golden: list[dict[str, Any]],
    available_queries: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for doc_id, text in sorted(corpus.items()):
            handle.write(
                json.dumps(
                    {
                        "chunk_id": doc_id,
                        "text": text,
                        "source": DATASET_NAME,
                        "doc_type": "stackoverflow_question",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    golden_payload = {
        "description": (
            "Deterministic global-corpus retrieval benchmark derived from the public "
            "MTEB StackOverflow duplicate-questions test split."
        ),
        "dataset": DATASET_NAME,
        "dataset_test_rows": available_queries,
        "seed": seed,
        "corpus_size": len(corpus),
        "test_cases": golden,
    }
    with (output_dir / "golden_test_set.json").open("w", encoding="utf-8") as handle:
        json.dump(golden_payload, handle, ensure_ascii=False, indent=2)


def build_bm25(
    output_dir: Path,
    corpus_ids: Sequence[str],
    corpus_texts: Sequence[str],
) -> tuple[BM25Indexer, QueryProcessor]:
    chunks = [
        Chunk(
            id=doc_id,
            text=text,
            metadata={
                "source_path": f"hf://{DATASET_NAME}/{doc_id}",
                "doc_type": "stackoverflow_question",
            },
        )
        for doc_id, text in zip(corpus_ids, corpus_texts)
    ]
    term_stats = SparseEncoder(min_term_length=1).encode(chunks)
    indexer = BM25Indexer(index_dir=str(output_dir / "bm25"))
    indexer.build(term_stats, collection="stackoverflow")
    return indexer, QueryProcessor()


def unit_normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def embedding_fingerprint(model: str, ids: Sequence[str]) -> str:
    payload = model + "\n" + "\n".join(ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embed_in_batches(
    embedding: Any,
    texts: Sequence[str],
    batch_size: int,
    progress_label: str,
) -> np.ndarray:
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = list(texts[start : start + batch_size])
        vectors.extend(embedding.embed(batch))
        completed = min(start + len(batch), total)
        if completed == total or completed % (batch_size * 10) == 0:
            print(f"{progress_label}: {completed}/{total}", flush=True)
    return unit_normalise(np.asarray(vectors, dtype=np.float32))


def load_or_create_embeddings(
    output_dir: Path,
    embedding: Any,
    model_name: str,
    corpus_ids: Sequence[str],
    corpus_texts: Sequence[str],
    golden: Sequence[dict[str, Any]],
    batch_size: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = output_dir / "dense_embeddings.npz"
    fingerprint = embedding_fingerprint(
        model_name,
        list(corpus_ids) + [item["query_id"] + ":" + item["query"] for item in golden],
    )

    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=False)
        cached_fingerprint = str(cached["fingerprint"].item())
        if cached_fingerprint == fingerprint:
            print(f"Using embedding cache: {cache_path}")
            return cached["corpus"], cached["queries"]

    corpus_matrix = embed_in_batches(
        embedding,
        corpus_texts,
        batch_size,
        "Corpus embeddings",
    )
    query_matrix = embed_in_batches(
        embedding,
        [item["query"] for item in golden],
        batch_size,
        "Query embeddings",
    )
    np.savez_compressed(
        cache_path,
        fingerprint=np.asarray(fingerprint),
        corpus=corpus_matrix,
        queries=query_matrix,
    )
    return corpus_matrix, query_matrix


def top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    k = min(top_k, scores.shape[0])
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    candidates = np.argpartition(-scores, k - 1)[:k]
    return candidates[np.argsort(-scores[candidates], kind="stable")]


class RetrievalBenchmark:
    def __init__(
        self,
        corpus_ids: Sequence[str],
        corpus_texts: Sequence[str],
        corpus_embeddings: np.ndarray,
        embedding: Any,
        bm25: BM25Indexer,
        query_processor: QueryProcessor,
        reranker: CrossEncoderReranker,
        retrieval_depth: int,
        rrf_k: int,
    ) -> None:
        self.corpus_ids = list(corpus_ids)
        self.corpus_texts = list(corpus_texts)
        self.text_by_id = dict(zip(corpus_ids, corpus_texts))
        self.corpus_embeddings = corpus_embeddings
        self.embedding = embedding
        self.bm25 = bm25
        self.query_processor = query_processor
        self.reranker = reranker
        self.retrieval_depth = retrieval_depth
        self.fusion = RRFFusion(k=rrf_k)

    def bm25_rank(self, query: str) -> list[RetrievalResult]:
        keywords = self.query_processor.process(query).keywords
        if not keywords:
            return []
        results = self.bm25.query(keywords, top_k=self.retrieval_depth)
        return [
            RetrievalResult(
                chunk_id=item["chunk_id"],
                score=float(item["score"]),
                text=self.text_by_id[item["chunk_id"]],
                metadata={"source": DATASET_NAME},
            )
            for item in results
        ]

    def dense_rank_from_vector(self, vector: np.ndarray) -> list[RetrievalResult]:
        vector = unit_normalise(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self.corpus_embeddings @ vector
        indices = top_indices(scores, self.retrieval_depth)
        return [
            RetrievalResult(
                chunk_id=self.corpus_ids[int(index)],
                score=float(scores[int(index)]),
                text=self.corpus_texts[int(index)],
                metadata={"source": DATASET_NAME},
            )
            for index in indices
        ]

    def dense_rank(self, query: str) -> list[RetrievalResult]:
        vector = np.asarray(self.embedding.embed([query])[0], dtype=np.float32)
        return self.dense_rank_from_vector(vector)

    def hybrid_rank_from_results(
        self,
        bm25_results: list[RetrievalResult],
        dense_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        return self.fusion.fuse(
            [dense_results, bm25_results],
            top_k=self.retrieval_depth,
        )

    def hybrid_rank(self, query: str) -> list[RetrievalResult]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            sparse_future = executor.submit(self.bm25_rank, query)
            dense_future = executor.submit(self.dense_rank, query)
            sparse = sparse_future.result()
            dense = dense_future.result()
        return self.hybrid_rank_from_results(sparse, dense)

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        candidates = [
            {
                "chunk_id": result.chunk_id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata,
            }
            for result in results
        ]
        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=self.retrieval_depth,
        )
        return [
            RetrievalResult(
                chunk_id=item["chunk_id"],
                score=float(item["rerank_score"]),
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            for item in reranked
        ]

    def hybrid_rerank(self, query: str) -> list[RetrievalResult]:
        return self.rerank(query, self.hybrid_rank(query))


def ranking_metrics(
    ranked_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> dict[str, float]:
    relevant = set(relevant_ids)
    top_k = list(ranked_ids[:k])
    hit_count = sum(1 for doc_id in top_k if doc_id in relevant)

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return {
        "hit_rate": 1.0 if hit_count else 0.0,
        "recall": hit_count / len(relevant) if relevant else 0.0,
        "mrr": reciprocal_rank,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def aggregate_metrics(rows: Sequence[dict[str, float]], k: int) -> dict[str, float]:
    if not rows:
        return {}
    return {
        f"hit_rate_at_{k}": statistics.fmean(item["hit_rate"] for item in rows),
        f"recall_at_{k}": statistics.fmean(item["recall"] for item in rows),
        f"mrr_at_{k}": statistics.fmean(item["mrr"] for item in rows),
        f"ndcg_at_{k}": statistics.fmean(item["ndcg"] for item in rows),
    }


def select_quality_gated_method(
    metrics: dict[str, dict[str, float]],
    k: int,
) -> dict[str, Any]:
    """Select the best strategy without allowing an offline quality regression.

    HitRate is the primary product metric, followed by MRR and nDCG.  The
    selected method is recorded in the report so an experimental fusion or
    reranker cannot silently replace a stronger baseline.
    """

    hit_key = f"hit_rate_at_{k}"
    mrr_key = f"mrr_at_{k}"
    ndcg_key = f"ndcg_at_{k}"
    selected = max(
        metrics,
        key=lambda name: (
            metrics[name][hit_key],
            metrics[name][mrr_key],
            metrics[name][ndcg_key],
        ),
    )
    baseline = "dense"
    rerank = "hybrid_rrf_rerank"
    reranker_accepted = (
        metrics[rerank][hit_key] >= metrics[baseline][hit_key]
        and metrics[rerank][mrr_key] >= metrics[baseline][mrr_key]
    )
    return {
        "primary_metric": hit_key,
        "tie_breakers": [mrr_key, ndcg_key],
        "baseline": baseline,
        "selected_method": selected,
        "reranker_accepted": reranker_accepted,
        "fallback_method": baseline if not reranker_accepted else rerank,
    }


def evaluate_rankings(
    benchmark: RetrievalBenchmark,
    golden: Sequence[dict[str, Any]],
    query_embeddings: np.ndarray,
    metric_k: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[str]]]]:
    metric_rows: dict[str, list[dict[str, float]]] = {
        "bm25": [],
        "dense": [],
        "hybrid_rrf": [],
        "hybrid_rrf_rerank": [],
    }
    rankings: dict[str, dict[str, list[str]]] = {}

    for index, test_case in enumerate(golden):
        query = test_case["query"]
        bm25_results = benchmark.bm25_rank(query)
        dense_results = benchmark.dense_rank_from_vector(query_embeddings[index])
        hybrid_results = benchmark.hybrid_rank_from_results(bm25_results, dense_results)
        reranked_results = benchmark.rerank(query, hybrid_results)

        per_method = {
            "bm25": bm25_results,
            "dense": dense_results,
            "hybrid_rrf": hybrid_results,
            "hybrid_rrf_rerank": reranked_results,
        }
        rankings[test_case["query_id"]] = {}
        for method, results in per_method.items():
            ids = [item.chunk_id for item in results]
            rankings[test_case["query_id"]][method] = ids[:metric_k]
            metric_rows[method].append(
                ranking_metrics(ids, test_case["expected_chunk_ids"], metric_k)
            )

        if (index + 1) % 50 == 0 or index + 1 == len(golden):
            print(f"Evaluation: {index + 1}/{len(golden)}", flush=True)

    return (
        {method: aggregate_metrics(rows, metric_k) for method, rows in metric_rows.items()},
        rankings,
    )


def measure_latency(
    benchmark: RetrievalBenchmark,
    golden: Sequence[dict[str, Any]],
    sample_count: int,
) -> dict[str, dict[str, float]]:
    methods: dict[str, Callable[[str], list[RetrievalResult]]] = {
        "bm25": benchmark.bm25_rank,
        "dense": benchmark.dense_rank,
        "hybrid_rrf": benchmark.hybrid_rank,
        "hybrid_rrf_rerank": benchmark.hybrid_rerank,
    }
    latencies: dict[str, list[float]] = {name: [] for name in methods}
    selected = list(golden[: min(sample_count, len(golden))])

    # Warm up local embedding and Cross-Encoder execution before timing.
    if selected:
        benchmark.dense_rank(selected[0]["query"])
        benchmark.hybrid_rerank(selected[0]["query"])

    for index, test_case in enumerate(selected):
        query = test_case["query"]
        for name, method in methods.items():
            started = time.perf_counter()
            method(query)
            latencies[name].append((time.perf_counter() - started) * 1000.0)
        if (index + 1) % 20 == 0 or index + 1 == len(selected):
            print(f"Latency: {index + 1}/{len(selected)}", flush=True)

    return {
        name: {
            "samples": len(values),
            "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95),
            "mean_ms": statistics.fmean(values) if values else 0.0,
        }
        for name, values in latencies.items()
    }


def build_report(
    args: argparse.Namespace,
    corpus_size: int,
    available_queries: int,
    golden: Sequence[dict[str, Any]],
    metrics: dict[str, dict[str, float]],
    latency: dict[str, dict[str, float]],
    embedding_model: str,
) -> dict[str, Any]:
    hit_key = f"hit_rate_at_{args.metric_k}"
    mrr_key = f"mrr_at_{args.metric_k}"
    dense_hit = metrics["dense"][hit_key]
    hybrid_hit = metrics["hybrid_rrf"][f"hit_rate_at_{args.metric_k}"]
    rerank_hit = metrics["hybrid_rrf_rerank"][f"hit_rate_at_{args.metric_k}"]
    bm25_mrr = metrics["bm25"][mrr_key]
    dense_mrr = metrics["dense"][mrr_key]
    hybrid_mrr = metrics["hybrid_rrf"][mrr_key]
    rerank_mrr = metrics["hybrid_rrf_rerank"][mrr_key]

    judged_pairs = sum(len(item["source_candidate_ids"]) for item in golden)
    return {
        "benchmark": {
            "dataset": DATASET_NAME,
            "dataset_test_rows": available_queries,
            "sample_seed": args.seed,
            "queries": len(golden),
            "global_corpus_documents": corpus_size,
            "labelled_query_candidate_pairs": judged_pairs,
            "retrieval_depth": args.retrieval_depth,
            "metric_k": args.metric_k,
            "evaluation_mode": "global corpus retrieval",
        },
        "models": {
            "embedding": embedding_model,
            "reranker": args.rerank_model,
        },
        "metrics": metrics,
        "latency": latency,
        "quality_gate": select_quality_gated_method(metrics, args.metric_k),
        "improvements": {
            f"dense_vs_bm25_hit_rate_at_{args.metric_k}_percentage_points": (
                dense_hit - metrics["bm25"][hit_key]
            ) * 100.0,
            f"dense_vs_bm25_mrr_at_{args.metric_k}_relative_percent": (
                ((dense_mrr / bm25_mrr) - 1.0) * 100.0 if bm25_mrr else 0.0
            ),
            f"hybrid_vs_dense_hit_rate_at_{args.metric_k}_percentage_points": (
                hybrid_hit - dense_hit
            ) * 100.0,
            f"rerank_vs_hybrid_hit_rate_at_{args.metric_k}_percentage_points": (
                rerank_hit - hybrid_hit
            ) * 100.0,
            f"rerank_vs_hybrid_mrr_at_{args.metric_k}_relative_percent": (
                ((rerank_mrr / hybrid_mrr) - 1.0) * 100.0 if hybrid_mrr else 0.0
            ),
            f"rerank_vs_dense_hit_rate_at_{args.metric_k}_percentage_points": (
                rerank_hit - dense_hit
            ) * 100.0,
            f"rerank_vs_dense_mrr_at_{args.metric_k}_relative_percent": (
                ((rerank_mrr / dense_mrr) - 1.0) * 100.0 if dense_mrr else 0.0
            ),
        },
    }


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    report_path = output_dir / "benchmark_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    benchmark = report["benchmark"]
    metrics = report["metrics"]
    latency = report["latency"]
    k = benchmark["metric_k"]
    lines = [
        "# Stack Overflow Technical Retrieval Benchmark",
        "",
        f"- Dataset: `{benchmark['dataset']}`",
        f"- Fixed seed: `{benchmark['sample_seed']}`",
        f"- Queries: **{benchmark['queries']}**",
        f"- Global corpus: **{benchmark['global_corpus_documents']}** documents",
        f"- Labelled candidate pairs: **{benchmark['labelled_query_candidate_pairs']}**",
        f"- Evaluation: global-corpus retrieval, metrics at K={k}",
        f"- Quality-gated selection: **{report['quality_gate']['selected_method']}**",
        f"- Reranker accepted: **{report['quality_gate']['reranker_accepted']}**",
        "",
        "| Method | HitRate@K | Recall@K | MRR@K | nDCG@K | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank"):
        row = metrics[method]
        timing = latency[method]
        lines.append(
            f"| {method} | {row[f'hit_rate_at_{k}']:.4f} | "
            f"{row[f'recall_at_{k}']:.4f} | {row[f'mrr_at_{k}']:.4f} | "
            f"{row[f'ndcg_at_{k}']:.4f} | {timing['p50_ms']:.2f} | "
            f"{timing['p95_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "Start the local Qwen3 embedding server, then run:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\benchmark_stackoverflow.py",
            "```",
            "",
            "The source dataset is public; generated corpus and labels retain its text for local "
            "evaluation only.",
        ]
    )
    (output_dir / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading public benchmark: {DATASET_NAME}")
    corpus, golden, available_queries = load_public_benchmark(args.queries, args.seed)
    corpus_ids = sorted(corpus)
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
    write_dataset_artifacts(
        output_dir,
        corpus,
        golden,
        available_queries,
        args.seed,
    )
    print(
        f"Prepared {len(golden)} queries, {len(corpus_ids)} global documents, "
        f"{sum(len(item['source_candidate_ids']) for item in golden)} labelled pairs"
    )

    print("Building BM25 index")
    bm25, query_processor = build_bm25(output_dir, corpus_ids, corpus_texts)

    settings = load_settings()
    embedding = EmbeddingFactory.create(settings)
    corpus_embeddings, query_embeddings = load_or_create_embeddings(
        output_dir=output_dir,
        embedding=embedding,
        model_name=settings.embedding.model,
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        golden=golden,
        batch_size=args.embedding_batch_size,
        force=args.force_embeddings,
    )

    import torch

    resolved_device = args.device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Cross-Encoder: {args.rerank_model} ({resolved_device})")
    from sentence_transformers import CrossEncoder

    cross_encoder = CrossEncoder(args.rerank_model, device=resolved_device)
    rerank_settings = replace(
        settings,
        rerank=RerankSettings(
            enabled=True,
            provider="cross_encoder",
            model=args.rerank_model,
            top_k=args.metric_k,
        ),
    )
    reranker = CrossEncoderReranker(rerank_settings, model=cross_encoder)

    benchmark = RetrievalBenchmark(
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        corpus_embeddings=corpus_embeddings,
        embedding=embedding,
        bm25=bm25,
        query_processor=query_processor,
        reranker=reranker,
        retrieval_depth=args.retrieval_depth,
        rrf_k=settings.retrieval.rrf_k,
    )

    metrics, rankings = evaluate_rankings(
        benchmark,
        golden,
        query_embeddings,
        args.metric_k,
    )
    with (output_dir / "top_k_rankings.json").open("w", encoding="utf-8") as handle:
        json.dump(rankings, handle, ensure_ascii=False)

    latency = measure_latency(benchmark, golden, args.latency_queries)
    report = build_report(
        args=args,
        corpus_size=len(corpus_ids),
        available_queries=available_queries,
        golden=golden,
        metrics=metrics,
        latency=latency,
        embedding_model=settings.embedding.model,
    )
    write_report(output_dir, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output_dir / 'benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

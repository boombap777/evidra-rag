#!/usr/bin/env python
"""Build and verify the evidence manifest for the Stack Overflow benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "data" / "benchmarks" / "stackoverflow"
DEFAULT_MANIFEST_PATH = DEFAULT_BENCHMARK_DIR / "benchmark_manifest.json"
DEFAULT_GATE_PATH = REPO_ROOT / "test_data" / "benchmark-gate.json"

ARTIFACT_PATHS = (
    "benchmark_report.json",
    "benchmark_report.md",
    "corpus.jsonl",
    "dense_embeddings.npz",
    "golden_test_set.json",
    "top_k_rankings.json",
    "bm25/stackoverflow_bm25.json",
)
SOURCE_ROOTS = ("src", "scripts", "tests", "config")
SOURCE_EXTRAS = (
    "DEV_SPEC.md",
    "LOCAL_DEPLOYMENT.md",
    "README.md",
    "pyproject.toml",
    "run-local.ps1",
    "uv.lock",
)


class ManifestError(RuntimeError):
    """Raised when evidence cannot be built or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"required evidence file is missing: {path}")
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def iter_source_files(repo_root: Path) -> Iterable[Path]:
    for root_name in SOURCE_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            yield path
    for relative in SOURCE_EXTRAS:
        path = repo_root / relative
        if path.is_file():
            yield path


def build_source_snapshot(repo_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(set(iter_source_files(repo_root))):
        relative = path.relative_to(repo_root).as_posix()
        evidence = file_evidence(path)
        entries.append({"path": relative, **evidence})

    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256(path + NUL + file_sha256 + LF)",
        "digest": digest.hexdigest(),
        "file_count": len(entries),
        "files": entries,
        "git_commit": None,
        "git_available": (repo_root / ".git").exists(),
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON evidence {path}: {exc}") from exc


def inspect_benchmark(benchmark_dir: Path) -> dict[str, Any]:
    report = _load_json(benchmark_dir / "benchmark_report.json")
    golden = _load_json(benchmark_dir / "golden_test_set.json")
    rankings = _load_json(benchmark_dir / "top_k_rankings.json")
    with (benchmark_dir / "corpus.jsonl").open("r", encoding="utf-8") as handle:
        corpus_rows = sum(1 for line in handle if line.strip())

    test_cases = golden.get("test_cases", []) if isinstance(golden, dict) else []
    benchmark = report.get("benchmark", {}) if isinstance(report, dict) else {}
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    latency = report.get("latency", {}) if isinstance(report, dict) else {}
    gate = report.get("quality_gate", {}) if isinstance(report, dict) else {}
    dense = metrics.get("dense", {})
    bm25 = metrics.get("bm25", {})
    dense_latency = latency.get("dense", {})

    checks = {
        "report_query_count_500": benchmark.get("queries") == 500,
        "report_corpus_count_14613": benchmark.get("global_corpus_documents") == 14613,
        "corpus_rows_14613": corpus_rows == 14613,
        "golden_cases_500": len(test_cases) == 500,
        "ranking_queries_500": isinstance(rankings, dict) and len(rankings) == 500,
        "metric_k_10": benchmark.get("metric_k") == 10,
        "dense_hit_rate_0838": dense.get("hit_rate_at_10") == 0.838,
        "dense_mrr_rounds_0529": round(float(dense.get("mrr_at_10", -1.0)), 3) == 0.529,
        "bm25_hit_rate_0718": bm25.get("hit_rate_at_10") == 0.718,
        "selected_method_dense": gate.get("selected_method") == "dense",
        "reranker_rejected": gate.get("reranker_accepted") is False,
        "dense_latency_samples_100": dense_latency.get("samples") == 100,
        "dense_p95_rounds_90_5_ms": round(float(dense_latency.get("p95_ms", -1.0)), 1) == 90.5,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ManifestError("benchmark content checks failed: " + ", ".join(failed))

    return {
        "checks": checks,
        "corpus_rows": corpus_rows,
        "golden_cases": len(test_cases),
        "ranking_queries": len(rankings),
        "resume_claim": {
            "queries": 500,
            "documents": 14613,
            "metric_k": 10,
            "bm25_hit_rate_at_10": bm25["hit_rate_at_10"],
            "bm25_mrr_at_10": bm25["mrr_at_10"],
            "dense_hit_rate_at_10": dense["hit_rate_at_10"],
            "dense_mrr_at_10": dense["mrr_at_10"],
            "dense_p95_ms": dense_latency["p95_ms"],
            "selected_method": gate["selected_method"],
        },
    }


def build_manifest(
    *,
    repo_root: Path,
    benchmark_dir: Path,
    llama_server: Path,
    embedding_model: Path,
    cpu: str,
    gpu: str,
    memory_gb: float,
) -> dict[str, Any]:
    inspection = inspect_benchmark(benchmark_dir)
    artifacts = {
        relative: file_evidence(benchmark_dir / relative)
        for relative in ARTIFACT_PATHS
    }
    report = _load_json(benchmark_dir / "benchmark_report.json")
    return {
        "schema_version": "stackoverflow-benchmark-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "offline public-dataset retrieval benchmark on one local machine",
        "source_snapshot": build_source_snapshot(repo_root),
        "benchmark": {
            **report["benchmark"],
            "command": (
                ".\\.venv\\Scripts\\python.exe scripts\\benchmark_stackoverflow.py "
                "--queries 500 --seed 20260901 --retrieval-depth 50 "
                "--metric-k 10 --latency-queries 100"
            ),
            "timing_protocol": {
                "timer": "time.perf_counter",
                "warmup_queries": 1,
                "measured_queries_per_method": 100,
                "concurrency": 1,
                "production_sla": False,
            },
        },
        "models": {
            "embedding": {
                "id": report["models"]["embedding"],
                "runtime": "llama.cpp OpenAI-compatible /v1/embeddings",
                "gguf": {"path": str(embedding_model.resolve()), **file_evidence(embedding_model)},
                "server": {"path": str(llama_server.resolve()), **file_evidence(llama_server)},
            },
            "reranker": {
                "id": report["models"]["reranker"],
                "accepted_by_quality_gate": report["quality_gate"]["reranker_accepted"],
            },
        },
        "hardware": {
            "cpu": cpu,
            "gpu": gpu,
            "memory_gb": memory_gb,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "artifacts": artifacts,
        "validation": inspection,
    }


def verify_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    benchmark_dir = manifest_path.parent
    checks: dict[str, bool] = {}

    try:
        inspection = inspect_benchmark(benchmark_dir)
        checks["benchmark_content"] = True
    except ManifestError:
        inspection = None
        checks["benchmark_content"] = False

    current_snapshot = build_source_snapshot(repo_root)
    recorded_snapshot = manifest.get("source_snapshot", {})
    checks["source_snapshot"] = (
        recorded_snapshot.get("digest") == current_snapshot["digest"]
        and recorded_snapshot.get("file_count") == current_snapshot["file_count"]
    )

    for relative, expected in manifest.get("artifacts", {}).items():
        path = benchmark_dir / relative
        checks[f"artifact:{relative}"] = path.is_file() and (
            expected.get("sha256") == sha256_file(path)
            and expected.get("size_bytes") == path.stat().st_size
        )

    embedding = manifest.get("models", {}).get("embedding", {})
    for label in ("gguf", "server"):
        expected = embedding.get(label, {})
        path = Path(str(expected.get("path", "")))
        checks[f"runtime:{label}"] = path.is_file() and (
            expected.get("sha256") == sha256_file(path)
            and expected.get("size_bytes") == path.stat().st_size
        )

    if inspection is not None:
        checks["resume_claim_matches_report"] = (
            manifest.get("validation", {}).get("resume_claim")
            == inspection.get("resume_claim")
        )

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": "stackoverflow-benchmark-gate-v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "current_source_snapshot_sha256": current_snapshot["digest"],
        "recorded_source_snapshot_sha256": recorded_snapshot.get("digest"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="write a fresh evidence manifest")
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    build.add_argument("--llama-server", type=Path, required=True)
    build.add_argument("--embedding-model", type=Path, required=True)
    build.add_argument("--cpu", required=True)
    build.add_argument("--gpu", required=True)
    build.add_argument("--memory-gb", type=float, required=True)

    verify = subparsers.add_parser("verify", help="verify all hashes and claims")
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    verify.add_argument("--output", type=Path, default=DEFAULT_GATE_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        payload = build_manifest(
            repo_root=REPO_ROOT,
            benchmark_dir=args.manifest.resolve().parent,
            llama_server=args.llama_server,
            embedding_model=args.embedding_model,
            cpu=args.cpu,
            gpu=args.gpu,
            memory_gb=args.memory_gb,
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"benchmark manifest written: files={payload['source_snapshot']['file_count']} "
            f"source={payload['source_snapshot']['digest']}"
        )
        return 0

    gate = verify_manifest(args.manifest.resolve(), REPO_ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": gate["passed"], "failed_checks": gate["failed_checks"]}))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

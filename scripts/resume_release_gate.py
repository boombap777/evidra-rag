#!/usr/bin/env python
"""Aggregate current-source evidence for the three RAG resume bullets."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_manifest import DEFAULT_MANIFEST_PATH, verify_manifest  # noqa: E402
from src.core.settings import load_settings  # noqa: E402
from src.mcp_server.tools.get_document_summary import TOOL_NAME as SUMMARY_TOOL  # noqa: E402
from src.mcp_server.tools.list_collections import TOOL_NAME as LIST_TOOL  # noqa: E402
from src.mcp_server.tools.query_knowledge_hub import TOOL_NAME as QUERY_TOOL  # noqa: E402

DEFAULT_JUNIT = REPO_ROOT / "test_data" / "j-final.xml"
DEFAULT_COVERAGE = REPO_ROOT / "test_data" / "coverage-final.json"
DEFAULT_OUTPUT = REPO_ROOT / "test_data" / "resume-release-gate.json"
MIN_TESTS = 1368
MIN_COVERAGE = 80.0

REQUIRED_TEST_NAMES = frozenset(
    {
        "test_dense_strategy_skips_sparse_and_rrf",
        "test_local_hash_embedding_is_deterministic_and_normalised",
        "test_successful_query_trace_records_strategy_candidates_and_final_ranks",
        "test_public_query_schema_rejects_out_of_range_top_k_before_execution",
        "test_close_is_idempotent_and_releases_client",
        "test_all_stages_have_method_field",
        "test_tools_call_query_knowledge_hub",
        "test_tools_call_list_collections",
        "test_tools_call_get_document_summary_missing",
        "test_ingest_simple_pdf",
        "test_init_default_metrics",
        "test_observed_benchmark_has_resume_counts_and_metrics",
    }
)


def read_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    names = {
        case.attrib.get("name", "")
        for suite in suites
        for case in suite.iter("testcase")
        if all(case.find(tag) is None for tag in ("skipped", "failure", "error"))
    }
    return {
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "test_names": names,
    }


def read_coverage(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["totals"]["percent_covered"])


def build_gate(junit_path: Path, coverage_path: Path, manifest_path: Path) -> dict[str, Any]:
    junit = read_junit(junit_path)
    coverage = read_coverage(coverage_path)
    benchmark = verify_manifest(manifest_path, REPO_ROOT)
    settings = load_settings(REPO_ROOT / "config" / "settings.yaml")
    missing_tests = sorted(REQUIRED_TEST_NAMES - junit["test_names"])
    tool_names = sorted({QUERY_TOOL, LIST_TOOL, SUMMARY_TOOL})

    checks = {
        "default_suite_no_failures": junit["failures"] == 0 and junit["errors"] == 0,
        "default_suite_size": junit["tests"] >= MIN_TESTS,
        "required_contract_tests": not missing_tests,
        "coverage_at_least_80": coverage >= MIN_COVERAGE,
        "benchmark_manifest_current": benchmark["passed"],
        "production_strategy_dense": settings.retrieval.strategy == "dense",
        "production_embedding_qwen3": settings.embedding.model == "Qwen3-Embedding-0.6B-Q8_0",
        "three_public_mcp_tools": tool_names
        == ["get_document_summary", "list_collections", "query_knowledge_hub"],
    }

    claims = {
        "RC1_RETRIEVAL_EVALUATION": all(
            checks[name]
            for name in (
                "benchmark_manifest_current",
                "production_strategy_dense",
                "production_embedding_qwen3",
            )
        ),
        "RC2_PLUGGABLE_LOCAL_DEPLOYMENT": all(
            checks[name]
            for name in (
                "default_suite_no_failures",
                "coverage_at_least_80",
                "production_embedding_qwen3",
            )
        ),
        "RC3_MCP_INGESTION_TRACE": all(
            checks[name]
            for name in (
                "default_suite_no_failures",
                "required_contract_tests",
                "three_public_mcp_tools",
            )
        ),
    }
    return {
        "schema_version": "rag-resume-release-gate-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "local/offline and public-dataset evaluation; not production SLA",
        "passed": all(checks.values()),
        "resume_ready": all(claims.values()) and all(checks.values()),
        "claims": {
            name: "SATISFIED" if passed else "PARTIAL"
            for name, passed in claims.items()
        },
        "checks": checks,
        "missing_required_tests": missing_tests,
        "junit": {key: value for key, value in junit.items() if key != "test_names"},
        "coverage_percent": coverage,
        "mcp_tools": tool_names,
        "source_snapshot_sha256": benchmark["recorded_source_snapshot_sha256"],
        "benchmark_gate": benchmark,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, default=DEFAULT_JUNIT)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gate = build_gate(args.junit.resolve(), args.coverage.resolve(), args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"resume_ready": gate["resume_ready"], "claims": gate["claims"]}))
    if args.require_ready and not gate["resume_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

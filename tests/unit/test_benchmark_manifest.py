"""Tests for the Stack Overflow benchmark evidence manifest."""

from pathlib import Path

import pytest

from scripts.benchmark_manifest import (
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_MANIFEST_PATH,
    ManifestError,
    inspect_benchmark,
    sha256_file,
    verify_manifest,
)


def test_sha256_file_is_content_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "evidence.txt"
    path.write_text("first", encoding="utf-8")
    first = sha256_file(path)
    path.write_text("second", encoding="utf-8")
    assert sha256_file(path) != first


@pytest.mark.benchmark_evidence
def test_observed_benchmark_has_resume_counts_and_metrics() -> None:
    inspection = inspect_benchmark(DEFAULT_BENCHMARK_DIR)
    assert inspection["corpus_rows"] == 14613
    assert inspection["golden_cases"] == 500
    assert inspection["resume_claim"]["dense_hit_rate_at_10"] == 0.838
    assert round(inspection["resume_claim"]["dense_p95_ms"], 1) == 90.5


def test_inspection_rejects_unrelated_directory(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        inspect_benchmark(tmp_path)


@pytest.mark.benchmark_evidence
def test_committed_manifest_matches_current_source_and_artifacts() -> None:
    gate = verify_manifest(DEFAULT_MANIFEST_PATH)
    assert gate["passed"], gate["failed_checks"]


def test_source_snapshot_detects_code_changes_without_benchmark(tmp_path: Path) -> None:
    from scripts.benchmark_manifest import build_source_snapshot

    source = tmp_path / "src"
    source.mkdir()
    (source / "example.py").write_text("value = 1\n", encoding="utf-8")
    first = build_source_snapshot(tmp_path)
    (source / "example.py").write_text("value = 2\n", encoding="utf-8")
    assert build_source_snapshot(tmp_path)["digest"] != first["digest"]


def test_manifest_verifier_rejects_missing_artifacts(tmp_path: Path) -> None:
    import json

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"artifacts": {"missing.json": {"sha256": "0" * 64}}}), encoding="utf-8"
    )
    result = verify_manifest(path, repo_root=tmp_path)
    assert not result["passed"]
    assert "artifact:missing.json" in result["failed_checks"]

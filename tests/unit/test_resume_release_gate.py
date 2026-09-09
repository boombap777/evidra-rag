"""Unit tests for the aggregate resume release gate helpers."""

import json
from pathlib import Path

from scripts.resume_release_gate import read_coverage, read_junit


def test_read_junit_collects_counts_and_names(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1">'
        '<testcase name="passed"/><testcase name="skipped"><skipped/></testcase>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    result = read_junit(path)
    assert result["tests"] == 2
    assert result["skipped"] == 1
    assert result["test_names"] == {"passed"}


def test_read_coverage_uses_coverage_py_total(tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(
        json.dumps({"totals": {"percent_covered": 86.25}}),
        encoding="utf-8",
    )
    assert read_coverage(path) == 86.25

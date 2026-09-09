"""Publication export has a bounded file set and refuses unsafe output targets."""

import runpy
from pathlib import Path

import pytest

PUBLICATION = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/publication.py"))


def test_export_excludes_runtime_files_and_does_not_copy_history(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for relative in ["README.md", ".env", ".coverage", "secret.log", "model.gguf", ".git/config"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sample", encoding="utf-8")
    target = tmp_path / "public"
    report = PUBLICATION["export"](root, target, {"project": "test", "include": ["."]})
    assert report["passed"]
    assert [row["path"] for row in report["files"]] == ["README.md"]
    assert (target / "PUBLICATION_MANIFEST.json").is_file()
    assert not (target / ".git").exists()
    assert (root / ".env").read_text() == "sample"


def test_export_refuses_existing_or_nested_target(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    policy = {"project": "test", "include": []}
    with pytest.raises(ValueError, match="outside"):
        PUBLICATION["export"](root, root / "public", policy)
    existing = tmp_path / "existing-public"
    existing.mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        PUBLICATION["export"](root, existing, policy)


def test_scan_rejects_token_without_printing_its_value(tmp_path):
    token = "sk-" + "A" * 48
    (tmp_path / "config.txt").write_text(token, encoding="utf-8")
    report = PUBLICATION["inventory"](tmp_path, {"project": "test", "include": ["*.txt"]})
    assert report["passed"] is False
    assert report["findings"][0]["rule"] == "api_token"
    assert token not in str(report)


def test_export_requires_declared_inputs(tmp_path):
    report = PUBLICATION["inventory"](
        tmp_path, {"project": "test", "include": [], "required": ["LICENSE"]}
    )
    assert report["findings"] == [{"path": "LICENSE", "rule": "required_file_missing"}]


def test_policy_cannot_escape_source_directory(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        PUBLICATION["selected_files"](tmp_path, {"include": ["../*"]})

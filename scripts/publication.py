"""Export an explicitly selected source release without staging or copying Git history.

The scan catches common token formats, not all possible secrets. Review the inventory.
Only new output directories outside this checkout are allowed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

DENIED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".auto-coder",
    ".publication",
    "node_modules",
    "htmlcov",
}
DENIED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".gguf",
    ".safetensors",
    ".bin",
    ".npz",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".pkl",
    ".pfx",
    ".key",
}
PATTERNS = {
    "api_token": re.compile(r"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{24,}"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})"),
    "aws_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s+[A-Za-z0-9+/=]{40,}"
    ),
}
PLACEHOLDER = re.compile(r"(?:placeholder|example|test|fake|dummy|redact)", re.I)


def selected_files(root: Path, policy: dict) -> list[Path]:
    selected = set()
    for pattern in policy["include"]:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("Unsafe publication include pattern")
        matches = [root] if pattern == "." else root.glob(pattern)
        for match in matches:
            if match.is_symlink():
                raise ValueError("Refusing symlink: " + match.relative_to(root).as_posix())
            candidates = []
            if match.is_file():
                candidates = [match]
            elif match.is_dir():
                for directory, dirs, files in os.walk(match, followlinks=False):
                    for name in dirs + files:
                        path = Path(directory) / name
                        if path.is_symlink():
                            raise ValueError(
                                "Refusing symlink: " + path.relative_to(root).as_posix()
                            )
                    dirs[:] = [name for name in dirs if name not in DENIED_PARTS]
                    candidates.extend(Path(directory) / name for name in files)
            for path in candidates:
                relative = path.relative_to(root)
                name = relative.as_posix()
                if any(part in DENIED_PARTS for part in relative.parts):
                    continue
                if path.suffix.lower() in DENIED_SUFFIXES:
                    continue
                if path.name.startswith((".coverage", ".tmp")):
                    continue
                if path.name.startswith(".env") and path.name != ".env.example":
                    continue
                if any(fnmatch.fnmatchcase(name, p) for p in policy.get("exclude", [])):
                    continue
                if not path.resolve().is_relative_to(root):
                    raise ValueError("File escapes repository: " + name)
                selected.add(relative)
    return sorted(selected, key=lambda p: p.as_posix())


def inventory(root: Path, policy: dict) -> dict:
    rows, findings = [], []
    for relative in selected_files(root, policy):
        data = (root / relative).read_bytes()
        name = relative.as_posix()
        rows.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if len(data) > 50 * 1024 * 1024:
            findings.append({"path": name, "rule": "over_release_size_budget"})
        # Only scan decodable text. Binary fixtures are listed and hashed, not certified.
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for rule, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if PLACEHOLDER.search(match.group()):
                    continue
                findings.append(
                    {"path": name, "line": text.count("\n", 0, match.start()) + 1, "rule": rule}
                )
    required = set(policy.get("required", []))
    present = {row["path"] for row in rows}
    for missing in sorted(required - present):
        findings.append({"path": missing, "rule": "required_file_missing"})
    return {
        "schema_version": "source-publication-v1",
        "scope": "selected working-tree files; no Git history or production certification",
        "project": policy["project"],
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "passed": not findings,
        "findings": findings,
        "files": rows,
    }


def export(root: Path, target: Path, policy: dict) -> dict:
    root, target = root.resolve(), target.resolve()
    if target == root or target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("Output must be a new directory outside the source checkout")
    if target.exists():
        raise ValueError("Output already exists; refusing to overwrite")
    report = inventory(root, policy)
    if not report["passed"]:
        raise ValueError("Publication scan failed: " + json.dumps(report["findings"]))
    target.mkdir(parents=True)
    for row in report["files"]:
        source = root / row["path"]
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Source changed during export: " + row["path"])
        destination = target / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (target / "PUBLICATION_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--zip", action="store_true", help="Also zip the new source directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    policy = json.loads((root / "scripts/publication.json").read_text(encoding="utf-8"))
    try:
        if args.zip and args.output is None:
            raise ValueError("--zip requires --output")
        if args.zip and Path(str(args.output) + ".zip").exists():
            raise ValueError("Zip already exists; refusing to overwrite")
        report = export(root, args.output, policy) if args.output else inventory(root, policy)
        if args.zip:
            shutil.make_archive(str(args.output.resolve()), "zip", args.output.resolve())
        summary = {key: value for key, value in report.items() if key != "files"}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()

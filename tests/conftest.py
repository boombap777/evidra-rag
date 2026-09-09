"""Pytest configuration and shared fixtures.

This module contains pytest configuration and fixtures that are shared
across all test modules.
"""

import os
import sys
from pathlib import Path

import pytest

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-benchmark-evidence", action="store_true", default=False,
        help="verify locally provisioned full benchmark, GGUF and runtime hashes",
    )
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run opt-in tests that call configured external/local model services",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_live = bool(config.getoption("--run-live"))
    skip_live = pytest.mark.skip(reason="live integration requires explicit --run-live")
    for item in items:
        if item.get_closest_marker("benchmark_evidence") and not config.getoption("--run-benchmark-evidence"):
            item.add_marker(pytest.mark.skip(reason="full benchmark artifacts require --run-benchmark-evidence"))
        path = Path(str(item.fspath))
        parts = {part.casefold() for part in path.parts}
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in parts:
            item.add_marker(pytest.mark.offline_integration)
        if item.get_closest_marker("live") is not None and not run_live:
            item.add_marker(skip_live)


@pytest.fixture(scope="session", autouse=True)
def offline_settings_profile(
    tmp_path_factory: pytest.TempPathFactory,
    pytestconfig: pytest.Config,
):
    """Make the default suite independent of credentials and model servers."""

    if pytestconfig.getoption("--run-live"):
        yield None
        return

    root = tmp_path_factory.mktemp("modular-rag-offline")
    chroma_dir = (root / "chroma").as_posix()
    traces_path = (root / "traces.jsonl").as_posix()
    settings_path = root / "settings.offline.yaml"
    settings_path.write_text(
        f"""
llm:
  provider: ollama
  model: disabled
  temperature: 0.0
  max_tokens: 64
embedding:
  provider: local_hash
  model: deterministic-contract-v1
  dimensions: 64
vector_store:
  provider: chroma
  persist_directory: "{chroma_dir}"
  collection_name: offline_test
retrieval:
  strategy: dense
  dense_top_k: 20
  sparse_top_k: 20
  fusion_top_k: 10
  rrf_k: 60
rerank:
  enabled: false
  provider: none
  model: none
  top_k: 5
evaluation:
  enabled: false
  provider: custom
  metrics: [hit_rate, mrr]
observability:
  log_level: INFO
  trace_enabled: true
  trace_file: "{traces_path}"
  structured_logging: true
ingestion:
  chunk_size: 1000
  chunk_overlap: 200
  splitter: recursive
  batch_size: 32
  chunk_refiner:
    use_llm: false
  metadata_enricher:
    use_llm: false
""".lstrip(),
        encoding="utf-8",
    )

    variable = "MODULAR_RAG_SETTINGS_PATH"
    previous = os.environ.get(variable)
    os.environ[variable] = str(settings_path)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        yield settings_path
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory path.
    
    Returns:
        Path to the project root directory.
    """
    return PROJECT_ROOT


@pytest.fixture
def sample_documents_dir(project_root: Path) -> Path:
    """Return the sample documents directory path.
    
    Args:
        project_root: The project root directory path.
        
    Returns:
        Path to the sample documents directory.
    """
    return project_root / "tests" / "fixtures" / "sample_documents"


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the config directory path.
    
    Args:
        project_root: The project root directory path.
        
    Returns:
        Path to the config directory.
    """
    return project_root / "config"

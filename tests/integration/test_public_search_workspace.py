"""The public UI exercises the real offline retrieval core and explicit error states."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.core.settings import load_settings
from src.observability.dashboard.pages import search_workspace
from src.observability.dashboard.services.query_service import search, seed_demo


@pytest.fixture
def demo_settings(tmp_path):
    settings = load_settings(Path(__file__).resolve().parents[2] / "config/settings.demo.yaml")
    return replace(
        settings,
        vector_store=replace(settings.vector_store, persist_directory=str(tmp_path / "chroma")),
    )


def test_demo_seeding_is_idempotent_and_search_has_sources(demo_settings):
    assert seed_demo(demo_settings) == seed_demo(demo_settings) == 3
    result = search(demo_settings, "MCP 工具 边界", "publication_demo", 3)
    assert len(result["results"]) == 3
    assert result["results"][0]["source"] == "mcp-demo.md"
    assert result["provider"] == "local_hash"
    assert result["elapsed_ms"] > 0
    assert result["stages"]
    assert not result["used_fallback"]


@pytest.mark.parametrize(
    "query,collection,top_k",
    [("", "demo", 3), ("x", "../escape", 3), ("x", "demo", True), ("x", "demo", 11)],
)
def test_search_rejects_invalid_inputs_before_storage(demo_settings, query, collection, top_k):
    with pytest.raises(ValueError):
        search(demo_settings, query, collection, top_k)


def test_seed_cannot_mutate_non_demo_collection(demo_settings):
    demo_settings = replace(
        demo_settings,
        vector_store=replace(demo_settings.vector_store, collection_name="private_collection"),
    )
    with pytest.raises(ValueError):
        seed_demo(demo_settings)


def test_search_page_seed_query_and_empty_input(demo_settings, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(search_workspace, "load_settings", lambda: demo_settings)

    def page():
        from src.observability.dashboard.pages.search_workspace import render

        render()

    app = AppTest.from_function(page, default_timeout=30).run()
    assert not app.exception
    app.button(key="seed_demo").click().run()
    assert not app.exception
    app.text_input[0].set_value("MCP 工具 边界")
    app.button[-1].click().run()
    assert not app.exception
    assert any("MCP" in item.value for item in app.text)
    app.text_input[0].set_value(" ")
    app.button[-1].click().run()
    assert any("先输入" in item.value for item in app.warning)
    assert not any("01 " in item.value for item in app.text)


def test_search_page_masks_backend_failure(demo_settings, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(search_workspace, "load_settings", lambda: demo_settings)

    def fail(*args):
        raise RuntimeError("private-backend-detail")

    monkeypatch.setattr(search_workspace, "search", fail)

    def page():
        from src.observability.dashboard.pages.search_workspace import render

        render()

    app = AppTest.from_function(page, default_timeout=30).run()
    app.text_input[0].set_value("MCP")
    app.button[-1].click().run()
    assert not app.exception
    assert any("检索暂不可用" in item.value for item in app.error)
    assert "private-backend-detail" not in str([item.value for item in app.error])

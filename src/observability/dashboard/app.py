"""Modular RAG Dashboard – multi-page Streamlit application.

Entry-point: ``streamlit run src/observability/dashboard/app.py``

Pages are registered via ``st.navigation()`` and rendered by their
respective modules under ``pages/``.  Pages not yet implemented show
a placeholder message.
"""

from __future__ import annotations

import streamlit as st


# ── Page definitions ─────────────────────────────────────────────────

def _page_search() -> None:
    from src.observability.dashboard.pages.search_workspace import render
    render()

def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render
    render()


# ── Navigation ───────────────────────────────────────────────────────

pages = [
    st.Page(_page_search, title="检索体验", default=True),
    st.Page(_page_overview, title="运行概览"),
    st.Page(_page_data_browser, title="资料库"),
    st.Page(_page_ingestion_manager, title="导入文档"),
    st.Page(_page_ingestion_traces, title="摄取记录"),
    st.Page(_page_query_traces, title="检索记录"),
    st.Page(_page_evaluation_panel, title="离线评估"),
]


def main() -> None:
    st.set_page_config(
        page_title="技术知识检索 · Modular RAG",
        page_icon="📊",
        layout="centered",
    )

    st.markdown("""<style>
      .stApp {background: #f7f8f6; color: #21312d;}
      .block-container {max-width: 980px; padding-top: 3rem;}
      h1 {letter-spacing: -.035em;}
      .stAppDeployButton {display: none;}
      div[data-testid='stVerticalBlockBorderWrapper'] {border-radius: 16px;}
      @media (prefers-reduced-motion: reduce) {* {animation: none !important;}}
    </style>""", unsafe_allow_html=True)
    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
else:
    # When run directly via `streamlit run app.py`
    main()

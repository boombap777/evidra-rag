"""Minimal search experience; snippets are evidence, not generated answers."""

import streamlit as st

from src.core.settings import load_settings
from src.observability.dashboard.services.query_service import search, seed_demo


def render() -> None:
    st.caption("MODULAR RAG / KNOWLEDGE WORKSPACE")
    st.title("让每次检索，都有据可查")
    st.markdown("输入技术问题，查看相关片段、来源与检索过程。这里展示检索证据，不生成或编造答案。")
    try:
        settings = load_settings()
    except Exception:
        st.error("配置暂不可用。请检查配置文件，或使用 --demo 启动离线演示。")
        return
    demo = settings.embedding.provider == "local_hash"
    st.info(
        "离线演示 · local_hash 仅验证链路，不代表 Qwen3 语义质量。"
        if demo
        else f"当前模型：{settings.embedding.model} · 策略：{settings.retrieval.strategy}"
    )
    if demo and settings.vector_store.collection_name == "publication_demo":
        if st.button("加载 3 篇内置样例", key="seed_demo"):
            try:
                with st.spinner("正在建立本地演示索引…"):
                    count = seed_demo(settings)
                st.success(f"{count} 篇样例已就绪；重复加载不会重复添加。")
            except Exception:
                st.error("样例加载失败，请检查本地存储配置与目录权限。")
    st.caption("试试：MCP 工具 边界 · RRF 排名 融合 · Trace 日志 来源")
    with st.form("knowledge_search"):
        query = st.text_input("你想查什么？", placeholder="例如：MCP 工具 边界", max_chars=2000)
        with st.expander("检索设置", expanded=False):
            collection = st.text_input("知识集合", value=settings.vector_store.collection_name)
            top_k = st.slider("最多返回片段数", 1, 10, 3)
        submitted = st.form_submit_button("检索资料", type="primary")
    if submitted:
        st.session_state.pop("search_result", None)
        if not query.strip():
            st.warning("先输入一个问题或关键词。")
        else:
            try:
                with st.spinner("正在检索并整理来源…"):
                    st.session_state.search_result = search(settings, query, collection, top_k)
            except ValueError:
                st.warning("请检查问题长度、集合名称和返回数量。")
            except Exception:
                st.error("检索暂不可用，请检查 Embedding 服务与索引配置。没有生成替代结果。")
    result = st.session_state.get("search_result")
    if result:
        st.divider()
        st.subheader("检索结果")
        st.caption(
            f"{len(result['results'])} 个片段 · {result['elapsed_ms']:.0f} ms · 来源于本次真实查询"
        )
        if result["used_fallback"]:
            st.warning("部分检索或重排路径不可用，本次结果使用了回退路径。")
        if not result["results"]:
            st.info("当前集合没有返回片段。首次体验请先加载样例，或在管理面板导入文档。")
        for index, row in enumerate(result["results"], 1):
            with st.container(border=True):
                st.text(f"{index:02d}  {row['title']}")
                st.caption(f"来源：{row['source']} · 相似度分数 {row['score']:.3f}（非置信度）")
                st.text(row["text"])
                with st.expander("片段标识"):
                    st.text(row["chunk_id"])
        with st.expander("查看检索过程"):
            st.json(
                {key: result[key] for key in ("strategy", "provider", "stages", "used_fallback")}
            )
    else:
        st.divider()
        st.caption("准备就绪。先加载样例，再输入关键词；真实文档可通过侧边栏的导入管理进入。")
    st.caption("本地调试 Trace 可能记录查询与片段，请勿输入私密信息。")

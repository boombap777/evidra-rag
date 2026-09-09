# 本地运行

本项目独立运行，不要求另外两个项目位于相邻目录。所有命令从当前源码根目录执行。

## 无模型演示（推荐先运行）

```shell
uv sync --extra dev --frozen
uv run --no-sync python scripts/start_dashboard.py --demo --port 8503
```

访问 http://127.0.0.1:8503/，加载自编样例后查询。该模式是 local_hash 合同演示，
不代表 Qwen3 的语义效果；演示索引与原有 Qwen3 索引分开。

## Qwen3 本地 Embedding

准备 llama.cpp 的 llama-server 与 Qwen3-Embedding-0.6B-Q8_0 GGUF，指定真实路径：

```powershell
.\run-local.ps1 -Mode dashboard -LlamaServer '<llama-server.exe路径>' -EmbeddingModel '<GGUF路径>'
```

也可设置 LLAMA_SERVER_PATH / RAG_EMBEDDING_MODEL_PATH 后省略对应参数。
脚本只启动自己的模型进程，结束时也只停止该进程。端口18282需空闲；不自动停止他人的服务。
Embedding 与 Dashboard 均绑定回环地址。服务模式下读取 config/settings.yaml 的 Qwen3配置。

## 摄取、查询与 MCP

先将 Mode 改为 embedding 并保持其终端运行，另开终端：

```shell
uv run --no-sync python scripts/ingest.py --path '<有权使用的PDF路径>' --collection technical_docs
uv run --no-sync python scripts/query.py --query '你的技术问题' --collection technical_docs --top-k 5 --no-rerank
uv run --no-sync mcp-server
```

MCP 使用 stdio，不是 REST URL；客户端必须配置本仓库为工作目录和独立 Python 环境。
三个工具是 query_knowledge_hub、list_collections、get_document_summary。
当前没有背景目录监控：文档摄取由用户主动触发。

测试、完整公开数据基准和许可说明以 [PUBLICATION.md](PUBLICATION.md) 为准。

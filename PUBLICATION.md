# 发布与复现

本项目是可公开的个人检索服务原型。源代码采用 MIT；第三方代码、数据和模型保留其原许可。

## 浏览器体验

前置：Python 3.12、uv。从仓库根目录执行：

```shell
uv sync --extra dev --frozen
uv run --no-sync python scripts/start_dashboard.py --demo --port 8503
```

打开 http://127.0.0.1:8503/，加载三个自编样例，检索 `MCP 工具 边界`。
该入口使用隔离的 `data/demo/chroma`，不会读取原有 Qwen3 索引。页面不生成答案，
不伪造引用；排序分数不是答案置信度。管理面板保留，未实现普通用户聊天产品。
只绑定本机回环地址；这是没有公网鉴权的本地管理工具，不应直接暴露到互联网。

## 测试与基准分离

```shell
uv run --no-sync pytest
```

默认测试使用临时索引及模拟依赖，不要求本机存在完整 benchmark 或模型。两个历史
基准制品验证用例由 `--run-benchmark-evidence` 显式开启；它们仍会严格验证源码、
数据和模型哈希，缺失时失败，不会因为默认测试跳过而被算作简历门禁成功。
`scripts/resume_release_gate.py` 只把实际通过的必需用例计入验收，不再把 skipped
用例仅因出现在 JUnit 中就视为通过。

## Qwen3 与完整公开数据基准

自行准备 Qwen3-Embedding-0.6B-Q8_0 GGUF 和 llama.cpp 的 `llama-server`（遵循各自许可）。
Windows 可以显式指定两个文件：

```powershell
.\run-local.ps1 -Mode embedding -LlamaServer '<llama-server.exe路径>' -EmbeddingModel '<Qwen3 GGUF路径>'
```

该服务只监听 `127.0.0.1:18282`，与 `config/settings.yaml` 相匹配；保持该终端运行。
另开终端，从仓库根目录运行：

```shell
uv sync --extra dev --extra benchmark --frozen
uv run python scripts/benchmark_stackoverflow.py --queries 500 --seed 20260901 --retrieval-depth 50 --metric-k 10 --latency-queries 100
```

脚本下载公开的 `mteb/stackoverflowdupquestions-reranking`，生成全局语料、标签、向量
和排名到 `data/benchmarks/stackoverflow`，并下载独立的 Cross-Encoder。它需要网络和
本地 Embedding 服务，不属于无模型演示。请查看脚本 `--help` 选择新输出目录，保留历史实验。
为新实验创建 manifest 时，用 `scripts/benchmark_manifest.py build --help` 显式提供
模型、运行时和硬件参数，再执行 `pytest --run-benchmark-evidence` 与完整简历门禁。
固定数字校验针对已报告的那次实验，不保证不同软件/硬件上的新实验恰好复现相同延迟。

Observed：500 查询、14,613 文档；Dense HitRate@10 83.8%，较 BM25 **高 12.0 个百分点**；
MRR@10 相对提升 32.9%。90.5 ms 是 100 条预热后本地查询的 P95，不是服务 SLA。
离线 local_hash 页面不展示这些数字为当前实时测量值。

## 发布内容

`python scripts/publication.py --output <不存在的外部目录>` 生成源码副本与逐文件哈希清单。
不含私有数据库、缓存、历史原始运行日志、凭据、Git 历史和大模型权重。
未确认来源的历史博主介绍 PDF 与 Vision 照片不在发布副本中；默认测试不需要它们。
Live Vision 测试需要自行提供有权使用的图片，默认跳过。
保留生成式测试夹具与自编演示知识；完整实验的获取方式与许可边界见本页及第三方说明。
导出扫描是有限的启发式检查，不是任意文本或历史记录绝无敏感信息的保证。

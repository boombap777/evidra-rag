# 检索实验：为什么默认使用 Dense

同一份测试数据上，Qwen3 Dense 的前十条命中率最高。RRF 和重排仍保留为实验选项，
但这次结果不足以支持把它们设为默认策略。

## 测试设置

- 数据来自公开的 [Stack Overflow 重复问题数据集](https://huggingface.co/datasets/mteb/stackoverflowdupquestions-reranking)。
- 从 2,992 条测试数据中按种子 `20260901` 抽取 500 条查询，构建 14,613 篇文档的全局语料。
- 在全局语料中检索，而不是只在每条查询附带的少量候选中排序；检索深度为 50，指标计算到第 10 条。
- 向量模型为 `Qwen3-Embedding-0.6B-Q8_0`；重排模型为 `cross-encoder/ms-marco-MiniLM-L-6-v2`。
- 延迟单独测量 100 条预热后的顺序查询，不是并发压力测试。

## 已记录的结果

| 方法 | HitRate@10 | MRR@10 | 本地 P95 |
| --- | ---: | ---: | ---: |
| BM25 | 0.718 | 0.3984 | 10.07 ms |
| Qwen3 Dense | 0.838 | 0.5295 | 90.48 ms |
| BM25 + Dense，经 RRF 融合 | 0.834 | 0.4810 | 90.06 ms |
| RRF 后再用 Cross-Encoder 重排 | 0.812 | 0.4914 | 267.66 ms |

HitRate@10 表示前十条中至少找到一条相关结果的查询占比；MRR@10 同时考虑第一条相关结果的位置。
Dense 相比 BM25 的命中率高 12.0 个百分点，MRR 相对提高 32.9%。
重排比 RRF 的 MRR 略高，但命中率下降，耗时也更长。

[汇总 JSON](benchmarks/stackoverflow-results.json) 保留原实验报告中的指标和测试设置。
它来自已有本地实验，本次 README 整理没有重新执行模型基准。完整语料、向量缓存、模型和逐查询结果不随源码发布。

## 复现

[发布说明](../PUBLICATION.md#qwen3-与完整公开数据基准)列出了模型准备、Embedding 服务启动和基准命令。
[基准脚本](../scripts/benchmark_stackoverflow.py)负责下载数据、构建语料并比较四条路径。

运行新实验时应使用新的输出目录，并记录实际模型、运行时和数据版本。结果不保证与这里完全相同，
尤其不能把单机延迟当作服务 SLA。无模型演示使用 `local_hash`，不复现本页的语义检索指标。

数据和模型的使用条件见[第三方说明](../THIRD_PARTY_NOTICES.md)。

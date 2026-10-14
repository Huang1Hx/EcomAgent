# MewHelp — 电商智能客服 Agent

MewHelp 是 EcomAgent 项目中的电商售后智能客服系统，覆盖订单与物流查询、政策 FAQ、
退款流程、知识补库、主题分类和评估观测。

系统采用确定性 Workflow 与 Agent 协作的架构：知识问答先检索并校验证据，退款流程先获取订单与政策；
其他业务由 Agent 按需调用受控工具获取事实并回答。

## 技术栈

FastAPI + LangGraph / LangChain + SQLAlchemy / MySQL + Milvus。

聊天、嵌入、重排三组上游各自直连，没有网关那一层。模型名和地址都在 `.env` 里配
(`CHAT_*` / `EMBED_*` / `RERANK_*` 三组)，换供应商、换模型不用改代码。

## 跑起来

```bash
cp .env.example .env      # 填 CHAT_* / EMBED_* / RERANK_* 三组
docker compose up -d      # MySQL
make seed                 # 灌业务测试数据
make dev                  # 依赖容器 + MCP :8101/:8102 + 应用 :8000
```

浏览器打开 <http://localhost:8000> 就是聊天页。

`make help` 可以查看带说明的常用命令及部分依赖提示；启动失败时先确认 `.env`、Docker 服务和模型上游配置。

## 代码怎么组织

| 位置 | 装的是什么 |
| - | - |
| `app/api/` | HTTP 入口。聊天、Agent、知识库录入、复核、验收页、成本看板 |
| `app/graph/` | LangGraph 那张图。`state` 状态、`nodes` 节点、`routing` 分流规则、`build` 组装 |
| `app/core/` | 单点能力。上游客户端、检索、重排、意图、指代、摘要、置信度、飞轮、可观测 |
| `app/kb/` | 知识怎么进库。切块、嵌入、双写 MySQL 与 Milvus、去重、从对话里挖问答对 |
| `app/tools/` | 工具系统。内置 `@tool`、MCP 客户端、注册表、统一执行引擎 |
| `app/db/` | 表模型与仓储 |
| `app/static/` | 前端页面。聊天、知识库录入、飞轮待审、观测与成本、主题分布、分类器验收 |
| `mcp_servers/` | 两台业务 MCP Server，物流和售后各一台，独立进程 |
| `sql/` | 数据库建表、迁移与种子数据，容器首启按文件名顺序自动执行 |
| `scripts/` | 建库、评估、微调这些离线活 |
| `dataset/` | 主题分类器的训练、验证和测试JSONL数据；后续统一从仓库根读取 |
| `docs/superpowers/` | 功能设计与实施计划 |

## 核心能力与验收命令

`make test` 跑全部单测，不打真实模型。下面这些要真服务在跑。

| 能力 | 主要内容 | 验收 |
| - | - | - |
| 对话入口 | 流式对话、结构化提取 | `make eval` |
| Agent 工具 | 五个业务工具、Function Calling | `make eval-agent` |
| 知识库 | 切块、嵌入、MySQL 与 Milvus 双写、对话挖知识 | `make kb-build` `make kb-vectorize` `make eval-retrieval` |
| 混合检索 | BM25、向量检索、RRF、重排、Query 改写 | `make smoke-rag` `make eval-rag` |
| 工作流 | LangGraph 分流、ReAct、退款 interrupt/resume | `make smoke-interrupt` `make eval-ch06` |
| 上下文 | 滑窗、摘要、前缀缓存 | `make eval-ch07` |
| 工具系统 | MCP 动态发现、统一执行引擎、审计日志 | `make eval-ch08` |
| 数据飞轮 | Langfuse、自检拒答、审核回写、成本统计 | `make langfuse-up` `make flywheel` `make eval-flywheel` `make cost-report` |
| 主题分类 | 语料治理、微调、阈值扫描、ONNX 推理服务 | `make ch10-corpus` `make ch10-train` `make ch10-eval` |

带说明的命令及其依赖提示可以通过 `make help` 查看。

## 端口

| 端口 | 是什么 |
| - | - |
| 8000 | 应用 |
| 8101 / 8102 | 业务 MCP Server，物流 / 售后 |
| 8110 | 主题分类器推理服务（`make classifier-up` 之后） |
| 3000 | Langfuse（`make langfuse-up` 之后） |
| 19530 | Milvus |

应用那几个页面：`/` 聊天、`/kb` 知识库录入、`/rag-eval` RAG 评估、`/agent-eval` Agent 评测、`/review` 飞轮待审、`/observability` 观测与成本、
`/topics` 主题分布、`/acceptance` 分类器验收。

# EcomAgent — 电商智能客服系统

EcomAgent 面向电商售后客服场景，覆盖订单与物流查询、政策 FAQ、
退款流程、知识补库、主题分类和评估观测。

系统采用确定性 Workflow 与 Agent 协作的架构：知识问答先检索并校验证据，退款流程先获取订单与政策；
其他业务由 Agent 按需调用受控工具获取事实并回答。

## 架构速览

在线主链由 LangGraph 固定执行“指代消解 → 九类意图识别 → 五个出口分流”。
知识问答走 BGE-M3 + BM25 召回、RRF 融合、BGE-Reranker 精排，再经四信号证据闸；
证据不足直接兜底并记录知识缺口。退款链先取本人订单、再查政策；商品浏览和订单查询先由
受控只读工具取得事实，Agent 只在这些必经步骤后组织回答。投诉和闲聊/其他分别走确定性出口。
三层上下文负责长对话事实保留；低置信、自检失败、用户负反馈是飞轮的三个入口，审核通过后才写回知识库。
17 类主题分类器是飞轮的旁路分析服务，不替代在线九类意图路由，也不负责决定客服回答。

## 当前验证状态

本分支保留了“功能已实现”和“发布评测已通过”的区别，完整状态见 [执行计划](newplan.md) 和
[评测口径](docs/evaluation.md)。截至 2026-10-19：

| 能力 | 当前证据 | 边界 |
| - | - | - |
| 分类器 | 现有 v1 测试集 161 条，micro-F1 0.9835；ONNX 可从私有 Release 校验恢复 | 不是新独立盲测；恢复需要仓库读取权限 |
| 混合检索 | 当前入库的 396 题报告中，282 道可答题的 Recall@5 0.881、MRR 0.593 | 与旧 300 题开发集不同；生成/拒答全量尚未完成 |
| 在线意图与 Agent | 意图 205/222；Agent 135/153 | 均未达到发布评测的全对门槛 |
| 飞轮与上下文 | 隔离飞轮闭环 46/50；上下文固定集 50/50 | 飞轮 4 题首轮未入池；固定集只检验上下文装配 |
| 工程回归 | 全量单测 680 passed、2 skipped | 发布聚合仍为 `failed`，报告对应较早代码 SHA |

评测题有 AI 辅助编写部分，用户确认曾人工抽检，但缺少可追溯的样本范围台账；
不能将整套题称为逐条人工审核。需补的字段和可定位样本见 [人工抽检记录模板](docs/eval-human-audit-template.md)。
历史简历指标不能自动作为当前分支成绩；面试介绍应使用本节带分母和边界的数字。
另有[逐题名次诊断报告](data/eval/reports/rag-retrieval-diagnosis.json)：其缓存重算值为
Recall@5 248.5/282（0.8812）、MRR 0.5919；它保留数据和命中缓存哈希及每题证据组名次，
与页面所读的另一轮报告略有波动，不是 RAG 端到端发布通过报告。旧新题集及缺失指标的区别见[评测口径](docs/evaluation.md)。

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

主题分类器已有私有 ONNX 发布包。具备本仓库 GitHub 读取权限并登录 `gh` 后，
运行 `uv sync --frozen --group ml && make classifier-bootstrap && make classifier-up`，
无需重新训练。需要从现有 `dataset/` 重新训练时，才运行 `make ch10-train`、
`make ch10-eval`、`make ch10-export`。
数据版本、模型位置与验收方式见 [分类器训练与恢复](docs/classifier-artifacts.md)。

模型包（未压缩约 409 MB）不进普通 Git；发布清单固定私有 Release 地址和 SHA256。
无仓库读取权限或未登录 `gh` 时，远程恢复会明确失败，不会自动重训。

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
| `dataset/` | 主题分类器的训练、验证和测试JSONL数据；脚本默认从仓库根读取 |
| `docs/`、`newplan.md` | 架构演进记录、评测口径、分类器恢复说明与当前任务状态 |

## 核心能力与验收命令

`make test` 跑全部单测，不打真实模型。下面这些要真服务在跑。

| 能力 | 主要内容 | 验收 |
| - | - | - |
| 对话入口 | 流式对话、结构化提取 | `make eval` |
| Agent 工具 | 商品浏览、本人订单、物流与 FAQ 等受控工具 | `make eval-agent` |
| 知识库 | 切块、嵌入、MySQL 与 Milvus 双写、对话挖知识 | `make kb-build` `make kb-vectorize` `make eval-retrieval` |
| 混合检索 | BM25、向量检索、RRF、重排、Query 改写 | `make smoke-rag` `make eval-rag` |
| 工作流 | LangGraph 分流、ReAct、退款 interrupt/resume | `make smoke-interrupt` `make eval-ch06` |
| 上下文 | 滑窗、摘要、前缀缓存 | `make eval-ch07` |
| 工具系统 | MCP 动态发现、统一执行引擎、审计日志 | `make eval-ch08` |
| 数据飞轮 | 自检拒答、隔离闭环、审核回写、成本统计 | `make flywheel` `make eval-flywheel-closure` `make cost-report` |
| 主题分类 | 语料治理、微调、阈值扫描、ONNX 推理服务 | `make ch10-corpus` `make ch10-train` `make ch10-eval` |
| 发布评测 | 分层数据合同、盲测、意图、Agent、飞轮、分类器、上下文聚合 | `make eval-release` |

带说明的命令及其依赖提示可以通过 `make help` 查看。
评测口径、数据审核状态和对外表述规则见 [发布评测口径](docs/evaluation.md)。

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

# 发布评测口径

本文件说明 `make eval-release` 如何汇总证据，以及当前报告能、不能支持哪些表述。
原始产物在 `data/eval/reports/`，单项报告仍由各自评测器生成。

## 一条命令看到全部状态

```bash
make eval-release                 # 按 contract→unit→rag→intent→agent→flywheel→classifier→context 顺序运行
make eval-release ARGS="--aggregate-only --skip-flywheel"   # 只聚合现有报告(不调用模型)
```

聚合器只读取各阶段报告，不重算指标、不改答案和阈值。阶段状态只有：
`passed`、`failed`、`provider_error`、`service_unavailable`、`not_run`、
`stale_report`、`invalid_report`、`blocked`。总状态取最严重的一项，
**未运行或上游错误永远不算通过**。

产物：

| 文件 | 内容 |
|---|---|
| `data/eval/reports/release-eval.json` | 每阶段状态、命令、报告路径、分子/分母与 Wilson 95% 区间 |
| `data/eval/reports/release-eval.md` | 同一结果的可读版本 |
| `data/eval/reports/project-evidence.json` | 七项能力的 `supported` / `partially_supported` / `unsupported` 映射 |
| `data/eval/reports/project-evidence.md` | 证据映射的可读版本 |

报告记录生成时间、Git SHA、数据集 SHA256、模型名与阈值。数据 SHA 与当前
`data/eval/*.jsonl` 不一致时标 `stale_report`；报告格式不满足对应阶段要求时标
`invalid_report`。

## 数据集与审核状态

- 旧 `tests/data/eval_ch04.jsonl` 300 题只作开发回归，报告标 `development_only`，
  不能用于独立测试结论。
- `data/eval/` 下的盲测、意图、Agent、飞轮和上下文集均为 AI 辅助编写或模板生成，
  `review_status=ai_assisted_unreviewed`、`reviewers=0`。它们能作为开发/回归证据，
  **不能写成人工审核或独立发布证据**；真实人工复核后才能改变表述。
- 置信度校准集与盲测集按锚点隔离。校准集只选阈值；盲测集只报告固定阈值下的
  误放率、误拦率及 95% 区间，不回调阈值。
- 飞轮闭环只在独立 MySQL 库与独立 Milvus 集合运行，不得写真实业务库。

## 阶段与当前证据边界

| 阶段 | 通过条件 | 说明 |
|---|---|---|
| 数据合同 | `tests/test_eval_dataset_contracts.py` 全部通过 | ID、字段、split、去重、支持数与 manifest 哈希 |
| 单元测试 | 全量 `pytest` 通过 | 2 条分类器 live 测试默认跳过，不算通过 |
| RAG 盲测 | 盲测数据匹配、检索与生成/拒答完整、无上游失败或跳过题 | 仅检索完整时阶段为 `not_run` 或 `provider_error`，检索能力只标 partially；499/429/超时不计成答错或正确拒答 |
| 在线意图 | 全部标注题判对且无上游错误 | 输出混淆矩阵与每类 P/R/F1 |
| Agent 多轮 | 全部题通过、无上游错误 | 校验工具、顺序、动作、结果与订单归属 |
| 飞轮闭环 | 隔离报告、数据 SHA 一致、写回链路非零 | 写回断言失败即整段失败 |
| 17 类分类器 | 报告权重 SHA 与校验后的 bundle 一致 | 现有 v1 测试集回归，不是新独立盲测 |
| 上下文固定集 | 确定性三层装配全部通过 | 不调模型；报告事实保留率、错误继承率、平均估 token 与装配 p95 |

## 对外表述规则

- 只引用 `release-eval.json` 中 `passed` 且数据/模型版本与当前仓库一致的分子/分母。
- 旧简历或历史章节中的准确率，只能在该阶段当前证据为 `passed` 时沿用；
  否则写成“历史结果，当前未复现”。
- “人工复核”只用于确实逐条复核过的样本；AI 辅助未审核数据不得使用该表述。
- 总状态不是 `passed` 时，写明未通过/阻塞阶段与原因，不写成“全链路验收完成”。

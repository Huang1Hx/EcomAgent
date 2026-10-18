# 发布评测报告

- 生成时间:2026-10-18T18:28:26
- 提交:1e51b391a64eddcbf69975e5f38814b6374b7ce6
- 总状态:invalid_report
- 数据口径:AI 辅助编写、未经真实人工审核(`review_status=ai_assisted_unreviewed`)

| 阶段 | 状态 | 指标 | 报告/错误 |
|---|---|---|---|
| 数据合同 | not_run | {} | 由主流程预置 |
| 单元测试 | not_run | {} | 由主流程预置 |
| RAG 盲测 | passed | {"recall_at_5": 0.881, "mrr": 0.593, "n_samples": 396, "refusal": {"threshold": 0.26, "answerable": {"correct": 116, "wrongly_refused": 166, "total": 282}, "absent": {"correct": 11 | /Users/huanghx/project/EcomAgent/data/ch04/reports/rag_eval.json |
| 在线意图 | not_run | {} |  |
| Agent 多轮 | invalid_report | {} | /Users/huanghx/project/EcomAgent/data/ch02/reports/agent_eval.json |
| 飞轮闭环 | blocked | {} | 未提供隔离飞轮环境或未运行 make eval-flywheel-closure |
| 17类分类器 | passed | {"micro_f1": 0.9835, "macro_f1": 0.9833, "n_samples": 161, "labels": 17} | /Users/huanghx/project/EcomAgent/data/ch10/reports/eval_report.json |
| 上下文固定集 | passed | {"fact_retention": {"numerator": 50, "denominator": 50, "rate": 1.0, "ci95": [0.9286, 1.0]}} | /Users/huanghx/project/EcomAgent/data/eval/reports/context_eval.json |

> 未运行、上游错误、服务不可用、报告与数据集 SHA 不一致均不计为通过。
> 旧 300 题仅为开发集;不用于本报告的独立测试结论。

# 发布评测报告

- 生成时间:2026-10-19T05:08:10
- 提交:b356b7b202b42ffce58b9759559f94dcf10f7e76
- 总状态:failed
- 数据口径:AI 辅助编写、未经真实人工审核(`review_status=ai_assisted_unreviewed`)

| 阶段 | 状态 | 指标 | 报告/错误 |
|---|---|---|---|
| 数据合同 | passed | {} | ......................                                                   [100%] 22 passed in 0.03s |
| 单元测试 | passed | {} | ........................................................................ [ 96%] ..........................                                               [100%] 672 passed, 2 skipped in 70.22s (0:01:10) |
| RAG 盲测 | not_run | {"recall_at_5": 0.881, "mrr": 0.593, "n_samples": 396, "refusal": {"threshold": 0.26, "answerable": {"correct": 116, "wrongly_refused": 166, "total": 282}, "absent": {"correct": 11 | /Users/huanghx/project/EcomAgent/data/ch04/reports/rag_eval.json |
| 在线意图 | failed | {"accuracy": {"numerator": 205, "denominator": 222, "rate": 0.9234, "ci95": [0.8808, 0.9516]}, "drift": {"numerator": 28, "denominator": 30, "rate": 0.9333, "ci95": [0.7868, 0.9815 | /Users/huanghx/project/EcomAgent/data/ch06/reports/intent_eval.json |
| Agent 多轮 | failed | {"passed": {"numerator": 135, "denominator": 153, "rate": 0.8824, "ci95": [0.8217, 0.9243]}, "tools": {"numerator": 103, "denominator": 138, "rate": 0.7464, "ci95": [0.6678, 0.8116 | /Users/huanghx/project/EcomAgent/data/ch02/reports/agent_eval.json |
| 飞轮闭环 | failed | {"closed_loop": {"numerator": 46, "denominator": 50, "rate": 0.92, "ci95": [0.8116, 0.9685]}, "first_refusal": {"numerator": 46, "denominator": 50, "rate": 0.92, "ci95": [0.8116, 0 | /Users/huanghx/project/EcomAgent/data/ch09/reports/flywheel_closure_eval.json |
| 17类分类器 | passed | {"micro_f1": 0.9835, "macro_f1": 0.9833, "n_samples": 161, "labels": 17} | /Users/huanghx/project/EcomAgent/data/ch10/reports/eval_report.json |
| 上下文固定集 | passed | {"fact_retention": {"numerator": 50, "denominator": 50, "rate": 1.0, "ci95": [0.9286, 1.0]}} | /Users/huanghx/project/EcomAgent/data/eval/reports/context_eval.json |

> 未运行、上游错误、服务不可用、报告与数据集 SHA 不一致均不计为通过。
> 旧 300 题仅为开发集;不用于本报告的独立测试结论。

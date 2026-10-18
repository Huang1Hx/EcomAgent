# 项目证据映射

| 能力 | 状态 | 报告 | 表述 |
|---|---|---|---|
| BGE-M3 + BM25 + RRF + BGE-Reranker | supported | /Users/huanghx/project/EcomAgent/data/ch04/reports/rag_eval.json | 四策略检索与盲测 Recall@5/MRR/答案覆盖 |
| 四信号置信度与拒答治理 | partially_supported | /Users/huanghx/project/EcomAgent/data/ch04/reports/rag_eval.json | 固定阈值下误放/误拦;阈值来自独立校准集 |
| 九类意图到五出口 Workflow + Agent | unsupported | ;/Users/huanghx/project/EcomAgent/data/ch02/reports/agent_eval.json | 九类意图、工具与订单归属校验 |
| 三层上下文、异步摘要、工具裁剪 | supported | /Users/huanghx/project/EcomAgent/data/eval/reports/context_eval.json | 上下文固定集事实保留率 |
| 三入口数据飞轮 | unsupported |  | 隔离环境闭环:拒答→入池→归并→批准→写回→再答 |
| 17类多标签 RoBERTa + ONNX | supported | /Users/huanghx/project/EcomAgent/data/ch10/reports/eval_report.json | 现有 v1 测试集回归;不是新独立盲测 |
| 架构决策可追溯性 | supported | data/eval/reports/release-eval.json | 按提交、报告和数据集合同追溯;仅当前证据 |

> 只有当前报告通过且数据/模型版本一致时才可标 supported;AI 辅助未审核集不能表述为人工复核证据。

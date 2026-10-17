# 评测数据说明(Task 3)

本目录存放分层评测集与 manifest。**所有数据只按实际证据表述**,不把 AI 辅助题写成人工审核,
也不把旧开发集重新包装成独立测试集。

## 来源与生成方式

| 文件 | 用途 / split | 来源 | 生成方式 | 审核状态 |
|---|---|---|---|---|
| `rag_dev.jsonl` | RAG 开发回归 / `dev` | `tests/data/eval_ch04.jsonl` 原 300 题 | 原题逐字保留,仅补 `source`/`split`/`review_status` 等血缘字段 | `legacy_unaudited`(无审核记录) |
| `rag_blind_test.jsonl` | RAG 盲测 / `test` | 知识库小节锚点 | 模板化改写问法(A/B/C/E/D 五桶) | `ai_assisted_unreviewed` |
| `intent_test.jsonl` | 在线九类意图 / `test` | 九类意图模板 + 跨轮漂移 + 边界表达 | 模板 + 语气变体,漂移带用户/客服历史 | `ai_assisted_unreviewed` |
| `agent_test.jsonl` | Agent 端到端 / `test` | 单轮、商品/订单/物流工具链、多轮状态 | 模板成题,逐题标注工具/顺序/动作/归属 | `ai_assisted_unreviewed` |
| `flywheel_closure_test.jsonl` | 飞轮闭环 / `test` | 知识缺口场景 | 模板成题,仅隔离环境执行 | `ai_assisted_unreviewed` |
| `context_test.jsonl` | 三层上下文 / `test` | 固定长会话事实集 | 事实插槽模板,50 段 | `ai_assisted_unreviewed` |
| `confidence_calibration.jsonl` | 阈值校准 / `calibration` | 知识锚点(奇偶锚点之一) | 单句 + 语气后缀 | `ai_assisted_unreviewed` |
| `confidence_blind_test.jsonl` | 固定阈值盲测 / `blind` | 另一组奇偶锚点 | 与校准集锚点、文本均不交叉 | `ai_assisted_unreviewed` |

生成脚本:`PYTHONPATH=. uv run python -m scripts.eval.build_suites`。
每个文件同目录的 `manifests/<name>.json` 记录样本数、桶/标签/审核支持数、SHA256、
生成方式、文本去重结果和 `independent_release_evidence=false`。

## 使用规则

- **旧 300 题只作开发集**:已参与过多轮提示词/检索调优,任何报告都必须标注
  `dataset_role=development_only`,不得作为独立测试分数的依据。
- **校准集与盲测集严格隔离**:`confidence_calibration.jsonl` 只用于选阈值;
  `confidence_blind_test.jsonl` 只在固定阈值下报告误放/误拦。不得用盲测结果回调答案或阈值。
- **未经人工审核不得宣称人工审核**:`review_status=ai_assisted_unreviewed`、`reviewers=0`
  的样本只能作开发/回归证据;要作为发布证据,须由真实人员逐条复核,更新
  `review_status`/`reviewers` 并重算 manifest 哈希。
- **不向真实业务库灌入评测样本**:飞轮闭环只在独立 MySQL 库与独立 Milvus 集合中执行
  (`scripts/eval_flywheel.py` 会拒绝与线上配置相同的环境)。
- 商品/订单题只验证客服查询与引导,不涉及支付或绕过订单归属校验。

## 复现与验证

```bash
PYTHONPATH=. uv run python -m scripts.eval.build_suites
uv run pytest tests/test_eval_dataset_contracts.py -q
```

数据合同覆盖:ID 唯一、必填字段、split 不交叉、文本精确去重、桶/标签支持数、manifest 哈希。

## 已知限制

- RAG 盲测的 B 桶由知识库型号锚点展开为 2 个问法;E/D 桶按锚点成题,
  因此各桶内问法数量分布不完全均衡,详见各 manifest 的 `support.bucket`。
- 意图边界表达不参与九类支持数,只验证不误归业务类。
- `context_test.jsonl` 只固定事实与轮数,不预置 AI 生成的逐轮对话,避免把生成结果当标准答案。

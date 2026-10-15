# MewHelp 证据、分类器、业务演示与会话一致性改造计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不把项目扩张成完整商城的前提下，补强评测证据，恢复17类旁路分类器，改善商品/订单演示体验，修复拒答后跨轮串答；Git根目录已安全迁移到 `EcomAgent/`。

**Architecture:** 保持在线九类意图路由、LangGraph 主链和17类离线主题分类器彼此独立。商城能力通过受控的演示商品目录和用户订单查询接口提供，客服只查询和引导，不承担真实支付下单；所有指标使用开发集、校准集、独立测试集分离后的报告。目录扁平化已在业务改造前独立完成，后续路径统一以仓库根为基准。

**Tech Stack:** Python 3.12、FastAPI、LangGraph/LangChain、MySQL、Milvus、ONNX Runtime、pytest、Docker Compose、uv。

**Spec:** 本文件的“需求基线”和“设计决策”即本轮获准的自包含规格；用户要求本轮不另写规格文件。

## 当前执行批次：先恢复分类器（2026-10-15）

用户已授权先写好计划，再在专用分支训练分类器。当前工作分支为 `feat/classifier-restoration`，起点 `0de8e19`；先执行分类器范围的基线、Task 4 和 Task 5。Task 1、2、3、6、8 的其他业务改造与新增评测数据后置。

- 使用原有 `dataset/{train,val,test}.jsonl`（2594/161/161），保持内容和划分不变。冻结哈希、17类顺序、逐类支持数和精确重叠检查，记录血缘信息缺失；测试结果标为现有 v1 数据集回归结果。
- 顺序固定为：现有数据核验与冻结 → 脚本统一数据目录 → 训练（仅train）与轮次/阈值选择（仅val）→ test评测 → ONNX导出及服务一致性 → 旁路联调 → 审核与提交。
- 训练默认沿用 RoBERTa 全参微调配置：学习率 `2e-5`、训练batch 16、最多8轮、早停耐心2、种子42；记录实际设备、依赖版本、基座revision、参数、代码SHA、耗时和最优轮次。资源不足时记录原因再调整batch，不根据test分数调参。
- 本批次完成本地模型资产清单、完整文件哈希、启动前校验及可复跑训练命令。对象存储/Release上传、下载器和新的独立测试数据作为后续交付，不阻塞本地训练和服务恢复。
- 训练/评测/ONNX产物使用同一数据版本和标签顺序；模型、tokenizer、阈值任一不匹配应非零退出。服务探活应暴露产物版本；启动失败不得仅打印文字后返回成功。
- 保护 `data/ch10/reports/golden_report.json`；旧评测报告先归档再生成本次报告。旁路验收使用测试数据库与已有样本的临时副本，禁止为了验证而向业务问题池灌入评测数据。
- 当前批次验收：可复跑训练、真实模型权重、test实测结果、Torch/ONNX/HTTP一致性、服务健康、旁路幂等测试、相关测试及完整回归；历史96.4%不作为硬性训练目标。

执行记录（仅凭实际结果勾选）：

- [x] C0：在 `feat/classifier-restoration` 分支冻结2594/161/161原数据；用户的 `golden_report.json` 哈希保持 `e45edf0…877a8`，未提交。
- [x] C1：统一根目录数据路径，记录逐文件SHA256、17类支持数、跨分割重叠及原数据血缘缺失；增加路径、哈希和失败启动测试。
- [x] C2：Apple MPS 训练8轮（最佳为第7轮），val选择阈值0.35；现有v1 test的micro-F1为0.9835、macro-F1为0.9833。ONNX对161条样本与Torch预测完全一致，见 `data/ch10/reports/{training_report,eval_report,export_report}.json`。
- [x] C3：本地模型和ONNX清单校验通过，HTTP健康接口暴露完整bundle身份；161条test通过真实HTTP与本地ONNX对照；独立MySQL测试库3条已有val样本首次写入、二次运行零写入，见 `tests/test_classifier_live.py`。
- [x] C4：独立代码审核后修复3项发现；最终含真实HTTP和隔离MySQL的完整测试 `531 passed`，本批次代码及报告单独提交到分类器分支。

本次代码审核提出的服务版本复用、失败导出保留旧包和阈值重演版本校验问题已补失败测试并修复。对外只能称“现有v1测试集回归”；新增盲测、远程模型分发和其他业务任务仍按下方未勾选任务推进。

## Global Constraints

- 2026-10-14已完成Git根目录迁移并把原外层数据集纳入仓库根 `dataset/`；除迁移所需Compose配置与本计划外，尚未修改业务代码和现有数据。
- 保留用户已有未提交内容：`data/ch10/reports/golden_report.json`，任何任务不得覆盖或顺手提交；本地非项目记录与部署草稿由 Git 排除规则保护。
- 不降低订单归属校验，不允许通过订单号枚举他人订单；未知商品必须继续返回未找到，不能生成虚假价格或库存。
- 不实现真实支付、库存扣减或地址收集；客服只提供商品查询、订单列表/详情、物流查询和跳转商城的建议动作。
- 在线九类意图路由继续由主链负责；17类RoBERTa仅服务飞轮问题的离线主题归类，不接管在线意图路由。
- 实施后必须保持本 EcomAgent 项目的七条技术主线：混合检索、四信号幻觉治理、九类到五出口、三层上下文、三入口飞轮、17类多标签分类器、评估驱动优化；商品/订单补强只能作为演示与验收支撑，不能把项目改造成商城交易系统而冲淡客服主线。
- 任何发布指标必须来自当前代码、当前模型和冻结数据集的重新运行，报告必须带 git SHA、数据集SHA256、知识库快照、模型版本和时间。
- 历史指标只作为待复验基线，不是开发验收的硬编码答案；如果独立测试结果不同，按当前报告更新数字与口径，禁止通过改答案、泄漏盲测集或放宽判定规则迎合预设结果。
- 每个任务按测试失败→最小实现→测试通过→独立审核→提交的顺序执行，不得把多个任务混入一次提交。

## Review Focus

- 确定性拒答、投诉和闲聊之后再进入Agent路径，历史中必须是一问一答，不能出现两个连续用户消息导致串答。
- “我的订单”只能列当前 `user_id` 名下订单；任意不存在或不归属订单仍不得泄露状态、金额和物流单号。
- “你们卖什么”“有什么玩具”“我要买幼猫粮”等表达应进入商品目录能力；“汽水”仍应明确当前不销售。
- 分类器数据、阈值、模型和ONNX必须来自同一版本；任一SHA不匹配时服务应拒绝启动并给出可操作错误。
- 目录扁平化后必须继续使用原MySQL/Milvus数据卷，不能因Compose项目名变化造成“数据消失”。

---

## 需求基线与已确认根因

1. **评测证据不足**
   - RAG主集有300题，但旧集已经参与多轮调优，不能继续当独立测试集。
   - Agent端到端评测只有7题；在线九类意图评测只有18题。
   - 飞轮评测复用RAG 300题，只验证指标趋势，没有验证“拒答→入池→归并→审核→写回→再次可答”。
   - 分类器数据实际为2594/161/161；测试集17类每类只有9–14个正例，且JSONL没有来源、审核状态等血缘字段。

2. **分类器只有代码和历史报告，没有模型资产**
   - 训练脚本读 `data/ch10/dataset/`，而现有数据位于仓库根 `dataset/`，当前没有接线。
   - `data/ch10/model/` 和 `data/ch10/onnx/` 被 `.gitignore` 排除，`classifier-up` 会因缺少 `model.onnx` 失败。

3. **商品和订单并非完全没有数据**
   - `data/catalog/products.json` 已有24个演示商品。
   - `app/tools/business.py` 会为每个非空用户生成稳定订单，并保证1001、2002可用。
   - 实际缺口是没有“列出我的订单”工具、通用商品浏览工具，以及购买诉求被九类意图误归“其他”；不应为此建设完整商城数据库。

4. **拒答后下一轮串答根因已定位**
   - `script_reply`、`complaint_reply`、`fallback_reply` 只写 `state.answer`。
   - `log_node` 把答复写入MySQL，却没有把对应 `AIMessage` 写回LangGraph checkpointer的 `messages`。
   - 下一轮进入 `main_agent` 时，消息窗口中会出现连续的历史 `HumanMessage`；实时日志已经出现“上一轮购买问题+本轮订单问题一起回答”的同型现象。

## 设计决策

- **评测扩充采用“旧集做开发、新集做盲测”**，而不是继续向已经调过参的300题中追加后宣称测试规模变大。
- **分类器模型可再生、产物可下载**：数据和训练配置进版本控制；大模型文件不直接进普通Git，使用带SHA256的artifact manifest和下载/校验脚本。
- **商城边界采用演示适配层**：补 `list_products`、`list_orders` 和商城跳转建议，不实现真实下单。
- **会话修复在 `log_node` 统一补齐确定性AI消息**：数据库落库成功后，只有确定性答复路径返回带 `db-<message_id>` 的 `AIMessage`；Agent路径已有AI消息，不重复追加。
- **目录扁平化不是零影响改名**：迁移时已重建`.venv`、显式固定并迁移Docker卷、保留Git历史和用户改动；后续开发不得重新引入 `EcomAgent/python/` 路径。

## 项目目标与发布验收矩阵

| 项目能力 | 计划对应任务 | 实施后必须保留的事实 | 发布条件 |
|---|---|---|---|
| BGE-M3 + BM25 + RRF + BGE-Reranker | Task 3、6 | 检索链路和五桶口径继续存在，不用商品/订单改造替换RAG主线 | 当前配置盲测报告给出Recall@5、MRR、答案覆盖；历史98.3%/89.4%/97.4%只在复验达到时继续使用 |
| 四信号置信度与拒答治理 | Task 3、6 | Top1、有效证据数、分差、关键条款命中四信号及生成前门控保留 | 独立校准集选阈值、独立测试集报告误放/误拦；“人工幻觉率”必须有逐条人工台账 |
| 九类意图到五个出口的Workflow + Agent | Task 1、2 | 九类和五出口不改名；知识、退款以及商品/订单事实获取都有不可跳过的确定性前置节点 | 图结构测试证明Agent无法绕过证据/订单/商品上下文；若仍存在直达Agent路径，发布说明必须限定适用范围 |
| 三层上下文、异步摘要、工具裁剪 | Task 1、3、6 | 原文层、半压缩层、摘要层和异步摘要机制保留 | 固定多轮集对比基线；只有事实保留率相对提升达到50%时才保留“提升50%” |
| 三入口数据飞轮 | Task 3、6 | 低置信、语义自检失败、用户负反馈三入口以及人工审核回写保留 | 端到端闭环集证明写回后可重新命中，并报告补库成功率和回归情况 |
| 17类多标签RoBERTa + ONNX | Task 4、5、6 | 分类器仍是飞轮旁路，不与在线九类意图混用 | 独立测试集报告micro-F1；只有达到96.4%才保留原数字。“轻量”需延迟、内存或量化证据，否则写“ONNX独立部署” |
| 架构决策可追溯性 | Task 0、8 | 用ADR、提交和验收报告记录 EcomAgent 的关键改造 | 对外说明只引用当前代码、提交记录和可复现验收结果，不使用无法追溯的主张 |

> 验收原则：项目能力和指标只按当前可复现证据表述，不能反向制造证据。

## 依赖与并行关系

```text
Task 7 目录扁平化（已于2026-10-14完成）
 ↓
Task 0 冻结基线
 ├─ Task 1 会话串答修复 ───────────────┐
 ├─ Task 2 商品/订单演示能力 ──────────┤  可并行组 A
 ├─ Task 3 新评测集设计与标注 ─────────┤
 └─ Task 4 分类器训练资产准备 ──────────┘

当前优先：Task 0（分类器基线）→ Task 4 → Task 5（原有v1数据训练/导出/服务验收）
Task 3新增数据独立后置，不是Task 5恢复服务的前置条件。
Task 1 + Task 2 + Task 3 + Task 5 → Task 6 全链路评测
Task 6 → Task 8 最终回归与文档
```

---

### Task 0: 冻结当前基线与保护用户改动

**Files:**
- Create: `data/eval/manifests/baseline.json`
- Preserve: `data/ch10/reports/golden_report.json`

**Interfaces:**
- Consumes: 当前Git状态、模型配置、知识库块数、各评测集文件。
- Produces: `baseline.json`，供后续报告比较，字段固定为 `git_sha`、`dirty_paths`、`models`、`kb_chunks`、`datasets`、`services`。

- [ ] **Step 1: 记录基线清单**
  仅记录密钥是否存在，不输出密钥内容；对所有评测JSONL记录实际样本数和SHA256。

- [ ] **Step 2: 运行当前可运行的单元测试**
  Run: `uv run pytest -q`
  Expected: 记录通过、失败、依赖服务错误三种数量，不把服务错误写成测试通过。

- [ ] **Step 3: 对现有在线流程做基线回放**
  固定同一 `user_id` 和新会话，回放商品浏览、我的订单、知识拒答、拒答后订单查询四组对话，保存请求与结构化响应。

- [ ] **Step 4: 审核并提交基线文件**
  只提交 `data/eval/manifests/baseline.json`，不得带入现有脏文件。

---

### Task 1: 修复确定性回复未进入LangGraph消息历史的问题

**Parallel:** 可与Task 2、Task 3、Task 4并行。

**Files:**
- Modify: `app/graph/nodes.py:419-430`
- Test: `tests/graph/test_log_node.py`
- Test: `tests/graph/test_turn_message_pairing.py`

**Interfaces:**
- Consumes: `repository.append_message(...) -> int` 返回的MySQL消息ID。
- Produces: `log_node(state) -> {"messages": [AIMessage]}`，仅在 `state.answer` 非空的确定性答复路径产生；消息ID为 `db-<mysql_message_id>`。

- [ ] **Step 1: 写失败单测 `test_log_node_adds_deterministic_reply_to_graph_history`**
  断言确定性 `answer` 落库后返回一条内容相同、ID为 `db-<id>` 的 `AIMessage`。

- [ ] **Step 2: 写失败单测 `test_log_node_does_not_duplicate_agent_reply`**
  状态已有Agent `AIMessage` 且 `answer` 为空时，断言 `log_node` 不再返回新消息。

- [ ] **Step 3: 写两轮回归测试 `test_fallback_then_business_turn_keeps_question_answer_pairs`**
  第一轮走 `fallback_reply`，第二轮走Agent；断言第二轮模型窗口中上一轮用户消息后紧跟拒答 `AIMessage`，且生成内容只响应第二轮问题。

- [ ] **Step 4: 运行测试确认失败**
  Run: `uv run pytest -q tests/graph/test_log_node.py tests/graph/test_turn_message_pairing.py`
  Expected: 新增断言因缺少确定性 `AIMessage` 而失败。

- [ ] **Step 5: 修改 `log_node` 做最小修复**
  保存数据库返回的消息ID；确定性答复返回带数据库锚点的 `AIMessage`，Agent路径维持现状。

- [ ] **Step 6: 验证流式接口没有重复输出**
  Run: `uv run pytest -q tests/graph/test_runtime.py tests/graph/test_log_node.py tests/graph/test_turn_message_pairing.py tests/test_chat_api.py`
  Expected: PASS；确定性文案仍只通过原有delta发送一次。

- [ ] **Step 7: 提交**
  Commit: `fix: preserve deterministic replies in conversation state`

---

### Task 2: 补齐商品浏览和“我的订单”演示能力

**Parallel:** 可与Task 1、Task 3、Task 4并行。

**Files:**
- Modify: `app/tools/catalog.py`
- Modify: `app/tools/builtin/orders.py`
- Modify: `app/tools/business.py`
- Modify: `app/core/prompts.py`
- Modify: `app/graph/routing.py`
- Modify: `app/graph/build.py`
- Modify: `app/graph/nodes.py`
- Modify: `app/graph/state.py`
- Modify: `app/static/index.html`
- Test: `tests/test_tools_mock.py`
- Test: `tests/tools/test_order_ownership.py`
- Test: `tests/graph/test_routing.py`
- Test: `tests/graph/test_build.py`
- Test: `tests/test_eval_agent_report.py`

**Interfaces:**
- Produces: `browse_products(category: str | None, limit: int = 10) -> list[dict]`。
- Produces tool: `list_products(category: str | None = None) -> {categories, products}`。
- Produces tool: `list_orders(user_id: InjectedToolArg) -> {orders}`。
- Produces node: `prepare_business_context(state: ConversationState) -> dict`，在 `business` 出口进入 `main_agent` 前确定性准备商品、订单或物流所需事实，并写入 `business_context`/`order_data`/`suggested_actions`。
- 保持: `query_order` 的用户归属校验和 `query_product` 的 `found=false` 语义。

- [ ] **Step 1: 写商品浏览失败测试**
  覆盖“你们卖什么”“有什么玩具”“猫粮有哪些”；断言返回真实目录数据，而不是走知识库拒答。

- [ ] **Step 2: 写订单列表失败测试**
  断言 `list_orders` 只返回注入用户自己的订单，并包含1001/2002；空身份返回空列表或明确身份错误。

- [ ] **Step 3: 写路由边界失败测试**
  “我要买幼猫粮”“帮我下单”“你们卖什么”应进入 `business`；“猫粮怎么选”仍走知识RAG；“汽水多少钱”进入目录查询并返回不销售。

- [ ] **Step 4: 实现两个只读工具并注册**
  不新增订单表，不新增支付写操作；`list_products`只读版本控制目录，`list_orders`只读当前mock适配层。

- [ ] **Step 5: 增加business确定性前置节点**
  将 `business` 条件边改为 `prepare_business_context -> main_agent`；商品查询先查目录，订单查询先校验归属或列出本人订单，物流查询先得到本人订单和物流单号。Agent只能组织已有事实或继续调用必要的物流工具，不能跳过前置检查。

- [ ] **Step 6: 修正Agent承诺边界**
  明确客服不能代下单、不能收集地址；查到商品后给出“前往商品页/商城下单”的建议，不说“我可以帮您下单”。

- [ ] **Step 7: 更新聊天页示例问题**
  增加“你们卖什么”“查看我的订单”“查订单1001物流”，让用户能发现可演示能力。

- [ ] **Step 8: 运行工具、路由、图结构和安全测试**
  Run: `uv run pytest -q tests/test_tools_mock.py tests/tools/test_order_ownership.py tests/graph/test_routing.py tests/graph/test_build.py tests/test_eval_agent_report.py tests/test_prompts.py`
  Expected: PASS，跨用户订单仍被拒绝，且图结构断言 `business` 不能直接进入 `main_agent`。

- [ ] **Step 9: 提交**
  Commit: `feat: add safe catalog browsing and order listing`

---

### Task 3: 重建分层评测数据与证据口径

**Parallel:** 数据设计与标注可与Task 1、Task 2、Task 4并行；最终跑分依赖Task 1、Task 2、Task 5。

**Files:**
- Create: `data/eval/README.md`
- Create: `data/eval/rag_dev.jsonl`
- Create: `data/eval/rag_blind_test.jsonl`
- Create: `data/eval/intent_test.jsonl`
- Create: `data/eval/agent_test.jsonl`
- Create: `data/eval/flywheel_closure_test.jsonl`
- Create: `data/eval/context_test.jsonl`
- Create: `data/eval/confidence_calibration.jsonl`
- Create: `data/eval/confidence_blind_test.jsonl`
- Create: `data/eval/manifests/*.json`
- Modify: `scripts/eval_ch04.py`
- Modify: `scripts/eval_intent.py`
- Modify: `scripts/eval_agent.py`
- Modify: `scripts/eval_flywheel.py`
- Modify: `scripts/calibrate_confidence.py`
- Test: `tests/test_eval_dataset_contracts.py`

**Interfaces:**
- 每条样本统一包含 `id`、`source`、`created_at`、`review_status`、`split`、任务专用字段。
- 每份数据manifest包含样本数、桶/标签支持数、SHA256、审核人次、生成方式和去重报告。

- [ ] **Step 1: 冻结旧300题为开发集**
  旧集只能用于调试与回归，报告中禁止再标为独立测试集。

- [ ] **Step 2: 新建RAG盲测集**
  新增至少300题，保持政策、型号、口语、库外、跨文档五桶各60题；不得从旧题简单同义改写，发布前不用于调参。

- [ ] **Step 3: 扩大在线意图集**
  从18题扩为至少180题（九类每类至少20题），另加至少45条跨轮漂移和边界表达；输出混淆矩阵和每类P/R/F1。

- [ ] **Step 4: 扩大Agent端到端集**
  从7题扩为至少120题：九类单轮90题、商品/订单/物流工具链15题、多轮状态15题；逐题校验工具、顺序、工具结果、回答和动作。

- [ ] **Step 5: 建立飞轮闭环集**
  至少50个知识缺口场景，验证第一次拒答、正确入池、归并去重、人工批准、MySQL/Milvus写回、第二次可答及原有题无回退。

- [ ] **Step 6: 建立上下文集**
  至少50段多轮对话，报告事实保留率、错误继承率、平均输入Token和p95延迟，不再把“50%预算”写成“提升50%”。

- [ ] **Step 7: 分离置信度校准集与盲测集**
  校准集只用于选择阈值；盲测集至少包含120条可答和120条库外/近似误导问题，只输出固定阈值下的误放率、误拦率及95%置信区间，不在盲测结果上再次调阈值。

- [ ] **Step 8: 加数据合同测试**
  检查ID唯一、split不交叉、文本精确去重、必填字段、桶支持数、标签支持数及manifest哈希。

- [ ] **Step 9: 提交数据和评测器**
  Commit: `test: add independent evaluation suites`

---

### Task 4: 分类器数据接线与可再生模型资产

**Parallel:** 训练资产准备可与Task 1、Task 2、Task 3并行。

**Files:**
- Canonical data: `dataset/{train,val,test}.jsonl`
- Modify: `scripts/ch10/train.py`
- Modify: `scripts/ch10/evaluate.py`
- Modify: `scripts/ch10/export_onnx.py`
- Modify: `scripts/ch10/scan_threshold_replay.py`
- Modify: `app/api/acceptance.py`
- Modify: `.gitignore`（只放行数据/模型清单，继续忽略大文件）
- Create: `scripts/ch10/artifact_manifest.py`
- Create: `scripts/ch10/fetch_model.py`
- Create: `data/ch10/artifact-manifest.example.json`
- Test: `tests/ch10/test_artifact_manifest.py`

**Interfaces:**
- 所有脚本接受 `--data-dir`，默认解析仓库根 `dataset/`，不再隐式依赖被忽略的 `data/ch10/dataset/`。
- `artifact-manifest.json`字段固定为 `dataset_sha256`、`base_model`、`train_config`、`threshold`、`model_sha256`、`onnx_sha256`、`metrics`、`artifact_uri`。
- `fetch_model.py --manifest <path>` 下载后必须校验SHA256，再原子写入 `data/ch10/onnx/`。

- [ ] **Step 1: 写路径与manifest失败测试**
  覆盖数据目录缺失、数据哈希不匹配、ONNX哈希不匹配和下载中断，断言明确失败且不留下半文件。

- [ ] **Step 2: 给训练/评测/导出脚本增加同一数据目录参数**
  train、val、test必须来自同一manifest版本。

- [ ] **Step 3: 恢复数据血缘**
  为数据增加旁挂manifest，记录现有2594/161/161的来源缺失事实；后续新样本必须记录真实、人工编写、合成或增强来源。

- [ ] **Step 4: 设计大文件分发方式**
  本批次先生成本地manifest及可复跑训练命令；对象存储/GitHub Release上传与下载器后置。普通Git中只提交manifest、代码和报告，不提交模型大文件。

- [ ] **Step 5: 提交资产管理代码**
  Commit: `build: make classifier artifacts reproducible`

---

### Task 5: 重训分类器、导出ONNX并打通旁路业务

**Depends on:** Task 4。

**Files:**
- Generate locally: `data/ch10/model/`
- Generate locally: `data/ch10/onnx/`
- Generate: `data/ch10/reports/eval_report.json`
- Generate: `data/ch10/reports/export_report.json`
- Generate: `data/ch10/artifact-manifest.json`
- Modify: `Makefile`
- Test: `tests/ch10/test_classifier_service.py`
- Test: `tests/ch10/test_classify_pool_e2e.py`

**Interfaces:**
- `make classifier-bootstrap`：已有且哈希正确则直接使用；否则按manifest下载；没有URI时给出 `make ch10-train && make ch10-export` 的明确指引。
- `make classifier-up`：启动前校验 `model.onnx`、`tokenizer.json`、`threshold.json` 和manifest。

- [x] **Step 1: 冻结现有分类器v1数据**
  核验原有2594/161/161及17类顺序、标签、分割交叉、重复与SHA256；保留原数据。扩大独立测试支持数的任务移交Task 3，在新一轮训练之前先冻结新数据版本。

- [x] **Step 2: 训练并记录环境**
  Run: `uv run --group ml python -m scripts.ch10.train --data-dir dataset`
  Expected: 生成模型、阈值和完整训练配置。

- [x] **Step 3: 在现有v1测试集评估**
  Run: `make ch10-eval`
  Expected: 输出micro/macro及每类P/R/F1、支持数、错误样本和置信区间。

- [x] **Step 4: 导出ONNX并校验一致性**
  Run: `make ch10-export`
  Expected: Torch/ONNX逐样本预测一致，报告文件SHA256、体积、CPU延迟和峰值内存。

- [x] **Step 5: 启动服务并验收旁路**
  Run: `make classifier-up && curl -fsS http://127.0.0.1:8110/healthz`
  使用隔离测试库和真实分类服务验证旁路写入与第二次运行幂等；不向业务问题池新增评测样本。

- [ ] **Step 6（后置）: 上传模型资产并回下载复验**
  从空的 `data/ch10/onnx/` 状态执行bootstrap，校验下载后服务可启动。

- [x] **Step 7: 提交报告和manifest**
  Commit: `feat: restore reproducible topic classifier service`

---

### Task 6: 运行全链路评测并建立发布门槛

**Depends on:** Task 1、Task 2、Task 3、Task 5。

**Files:**
- Create: `scripts/run_release_eval.py`
- Create: `data/eval/reports/release-eval.json`
- Create: `data/eval/reports/release-eval.md`
- Create: `data/eval/reports/project-evidence.json`
- Create: `data/eval/reports/project-evidence.md`
- Modify: `Makefile`
- Test: `tests/test_release_eval_report.py`

**Interfaces:**
- `make eval-release`按固定顺序运行：数据合同→单测→RAG盲测→意图→Agent多轮→飞轮闭环→分类器→上下文。
- 报告明确区分 `passed`、`failed`、`provider_error`、`service_unavailable`、`not_run`，不把缺服务算成通过。
- `project-evidence.json`逐条对应上方七项项目能力，状态只能是 `supported`、`partially_supported`、`unsupported`，并附报告路径、分子、分母、配置和建议发布表述。

- [ ] **Step 1: 写聚合报告失败测试**
  确认任何子任务未运行或报告SHA不匹配时总状态不能是通过。

- [ ] **Step 2: 实现只聚合、不重算口径的发布评测器**
  每项指标保留分子/分母和95%置信区间，不只保存百分比。

- [ ] **Step 3: 运行全链路评测**
  所有服务启动后执行 `make eval-release`；修复失败时回到所属Task，不在聚合器里绕过。

- [ ] **Step 4: 人工复核忠实度失败案例**
  保存逐条处置状态；只有实际复核过的样本才能表述为“人工复核”。

- [ ] **Step 5: 生成项目证据映射**
  对Recall@5/MRR/覆盖率、误放/误拦、上下文提升、飞轮闭环和分类器micro-F1逐项读取当前发布报告；任何未运行、旧模型或旧数据报告一律不能标为 `supported`。

- [ ] **Step 6: 提交发布报告**
  Commit: `test: add release evaluation gate`

---

### Task 7: 将Git仓库从 `EcomAgent/python/` 安全扁平化到 `EcomAgent/`（已完成）

**Depends on:** 已按用户要求提前执行，为后续任务统一路径基准。

**Parallel:** 不可并行。执行时必须停止应用、MCP、分类器、后台评测和所有会写项目文件的进程。

**Migration result:** Git根已迁移到 `EcomAgent/.git`；`.venv`已重建；Compose卷已迁移为显式的 `mewhelp-*` 名称，旧卷和离线快照暂留作回退。

**Files:**
- Move: `python/.git` → `.git`
- Move: `python/*`及受控隐藏文件 → 仓库根
- Keep: `dataset/` at repository root
- Recreate: `.venv/`
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Modify: any scripts/docs containing old absolute cwd

**Interfaces:**
- 新Git根固定为 `/Users/huanghx/project/EcomAgent`。
- Docker数据卷使用显式稳定名称，不再依赖工作目录推导Compose项目名。

- [x] **Step 1: 建立可恢复快照**
  记录Git SHA和脏文件；备份MySQL，记录四个Docker卷名并为Milvus卷制作可恢复快照；确认备份可读。

- [x] **Step 2: 停止全部服务**
  Run: `make dev-down && make classifier-down && docker compose stop`
  Expected: 8000、8101、8102、8110停止监听，容器停止但卷存在。

- [x] **Step 3: 先固定Compose卷身份**
  在移动目录前使Compose显式指向现有数据卷，并用 `docker compose config` 验证移动前后解析到相同卷。

- [x] **Step 4: 移动仓库内容并保留所有用户文件**
  不覆盖已存在的 `dataset/` 和父目录文件；单独清点并迁移日志和本地配置，本地记录继续由忽略规则保护。

- [x] **Step 5: 删除并重建虚拟环境**
  旧`.venv`不搬迁；在新根执行 `uv sync`，避免脚本shebang继续指向旧路径。

- [x] **Step 6: 验证Git身份和差异**
  `git rev-parse --show-toplevel`必须返回 `/Users/huanghx/project/EcomAgent`；除计划内文件外，移动前后的已跟踪内容哈希一致。

- [x] **Step 7: 启动并核对原数据**
  MySQL行数、Milvus集合及向量数必须与Task 0基线一致；不一致立即停止并从快照恢复。

- [x] **Step 8: 提交路径迁移**
  Commit: `chore: flatten repository root`

---

### Task 8: 最终回归、文档与项目证据更新

**Depends on:** Task 6；Task 7已完成。

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `newplan.md`（仅勾选实际完成项、记录最终报告路径）
- Create: `docs/evaluation.md`
- Create: `docs/classifier-artifacts.md`

**Interfaces:**
- 新用户从根目录依次执行 `uv sync`、配置`.env`、`docker compose up -d`、建库、`make dev`即可运行主业务。
- 分类器通过 `make classifier-bootstrap && make classifier-up`可恢复，不依赖开发者本机遗留文件。

- [ ] **Step 1: 从干净环境演练部署文档**
  不使用已有`.venv`、模型缓存或数据库状态完成一次部署验收。

- [ ] **Step 2: 验证四个用户问题全部关闭**
  评测证据可追溯；分类器可恢复；商品/订单能正常演示且安全边界不变；拒答后下一轮不串答。

- [ ] **Step 3: 更新项目介绍和发布指标**
  保留“Workflow + Agent、混合检索、幻觉治理、三层上下文、数据飞轮、17类旁路分类器”主线；数字只引用 `release-eval.json` 中当前、可复现、带分母和置信区间的指标，对外统一表述为本 EcomAgent 项目的实现与验收结果。

- [ ] **Step 4: 最终审核与提交**
  Run: `git status --short && git diff --check && uv run pytest -q && make eval-release`
  Expected: 仅存在用户明确保留的未提交文件；测试和发布评测状态如实记录。

## 执行建议

- 当前先执行分类器范围的Task 0 → Task 4 → Task 5；不等待新增评测数据。
- 分类器恢复后执行Task 1，再推进Task 2与Task 3；Task 1修复完成前不运行多轮发布评测。
- Task 7已先行完成；后续所有命令、文档和代码路径都以 `EcomAgent/` 为仓库根。
- 当前用户已授权分类器实现、训练与服务验收；其他业务按后续批次实施。

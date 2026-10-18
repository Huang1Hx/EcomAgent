"""Task 3 评测器:意图混淆矩阵、Agent 工具/顺序/归属、置信度固定阈值口径。"""
import pytest

from scripts.eval_agent import evaluate_case, load_cases
from scripts.eval_ch04 import refusal_metrics
from scripts.eval_intent import confusion, prf


def test_intent_confusion_matrix_per_class():
    gold = ["物流", "物流", "订单", "订单", "售后"]
    pred = ["物流", "订单", "订单", "订单", "投诉"]
    table = confusion(pred, gold, ["物流", "订单", "售后", "投诉"])
    assert table["物流"] == {"物流": 1, "订单": 1, "售后": 0, "投诉": 0}
    m = prf(table)
    assert m["物流"]["precision"] == 1.0 and m["物流"]["recall"] == 0.5
    assert m["订单"]["precision"] == 0.6667 and m["订单"]["recall"] == 1.0
    assert round(m["售后"]["f1"], 4) == 0.0


def test_agent_tools_only_checked_when_sample_requests_tools():
    """checks 未要求 tools 时,不必因为模型直接回答而判失败;答案约束仍生效。"""
    from scripts.eval_agent import evaluate_case

    sample = {"id": "AGT-004", "query": "订单 1001 多少钱", "tools_required": ["query_order"],
              "checks": ["answer_nonempty"], "order_scope": "own",
              "answer_any": ["1739"]}
    body = {"answer": "订单 1001 的金额是 1739 元。", "suggested_actions": [],
            "tool_calls": [], "tool_results": []}
    case = evaluate_case(sample, body)
    assert case["checks"]["tools"] is True
    assert case["passed"] is True


def test_agent_order_selection_interrupt_is_pending_not_empty_answer():
    """缺订单号时图停在 select_order:这是等待用户点选,不是空答复失败。"""
    from scripts.eval_agent import pending_interaction

    case = pending_interaction({"id": "AGT-041", "kind": "single", "query": "我要退货"},
                               {"conversation_id": 1, "answer": "", "tool_calls": [],
                                "tool_results": [], "suggested_actions": [],
                                "interrupt": {"type": "select_order",
                                              "orders": [{"order_id": "1001"}]}})
    assert case["passed"] is True
    assert case["checks"]["interaction_pending"] is True
    assert case["checks"]["answer_nonempty"] is True


def test_agent_multi_turn_all_interrupts_is_pending_not_failure():
    from scripts.eval_agent import multi_turn_result

    case = multi_turn_result({"id": "M1", "turns": ["我要退货", "1001"]},
                             [{"answer": "", "interrupt": {"type": "select_order"}},
                              {"answer": "", "interrupt": {"type": "select_order"}}], None)
    assert case["passed"] is True
    assert case["checks"]["all_turns_interrupted"] is True
    assert case["checks"]["answer_nonempty"] is False


def test_agent_toolchain_checks_order_and_result():
    sample = {"id": "x", "query": "我的订单和其中一单的物流",
              "tools_required": ["list_orders", "query_logistics"], "tools_in_order": True,
              "order_scope": "own", "checks": ["tools", "tool_order", "tool_result",
                                               "answer_nonempty", "order_scope"]}
    body = {"answer": "已发货", "suggested_actions": [],
            "tool_calls": [{"name": "list_orders"}, {"name": "query_logistics"}],
            "tool_results": [{"name": "list_orders", "content": '{"orders": [{"id": 1001}]}'},
                             {"name": "query_logistics", "content": '{"order_id": 1001}'}]}
    ok = evaluate_case(sample, body)
    assert ok["passed"] is True and ok["checks"]["tool_order"] is True
    reversed_ = dict(body, tool_calls=[{"name": "query_logistics"}, {"name": "list_orders"}])
    assert evaluate_case(sample, reversed_)["passed"] is False
    foreign = dict(body, tool_results=[{"name": "list_orders", "content": '{"orders": []}'},
                                       {"name": "query_logistics",
                                        "content": '{"status": "not_found"}'}])
    assert evaluate_case(sample, foreign)["checks"]["order_scope"] is False


def test_agent_empty_tools_must_stay_empty():
    sample = {"id": "x", "query": "今天天气怎么样", "tools_required": []}
    assert evaluate_case(sample, {"answer": "不在业务范围", "tool_calls": [],
                                  "tool_results": [], "suggested_actions": []})["passed"]
    assert not evaluate_case(sample, {"answer": "好", "tool_calls": [{"name": "list_orders"}],
                                      "tool_results": [], "suggested_actions": []})["passed"]


def test_agent_suite_has_required_composition():
    cases = load_cases()
    kinds = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    assert kinds["single"] >= 90 and kinds["toolchain"] >= 15 and kinds["multi"] >= 15


def test_refusal_metrics_use_fixed_threshold():
    samples = [{"id": "a", "should_refuse": False}, {"id": "b", "should_refuse": True}]
    hits = {"a": [], "b": []}
    m = refusal_metrics(samples, hits, threshold=0.26)
    assert m["answerable"]["wrongly_refused"] == 1
    assert m["absent"]["correct"] == 1


def test_milvus_search_defaults_resolve_collection_at_call_time():
    """隔离评测会在运行时切换 milvus_client.COLLECTION;默认参数不能冻在导入时的名字上。"""
    import inspect

    from app.kb import milvus_client

    for fn in (milvus_client.dense_search, milvus_client.bm25_search,
               milvus_client.hybrid_search, milvus_client.count):
        assert inspect.signature(fn).parameters["collection"].default is None


def test_eval_ch04_concurrency_is_configurable(monkeypatch):
    """学校上游与 CLI 共用额度:默认单路,配置必须控制真实信号量。"""
    from scripts import eval_ch04

    monkeypatch.delenv("RAG_RETR_CONCURRENCY", raising=False)
    monkeypatch.delenv("RAG_GEN_CONCURRENCY", raising=False)
    assert eval_ch04.eval_concurrency() == (1, 1)
    monkeypatch.setenv("RAG_RETR_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_GEN_CONCURRENCY", "2")
    assert eval_ch04.configure_concurrency() == (1, 2)
    assert eval_ch04._retr_sem._value == 1
    assert eval_ch04._gen_sem._value == 2


def test_eval_ch04_invalid_limits_fail_before_running(monkeypatch):
    from scripts import eval_ch04

    monkeypatch.setenv("RAG_RETR_CONCURRENCY", "0")
    with pytest.raises(ValueError):
        eval_ch04.configure_concurrency()
    monkeypatch.setenv("RAG_RETR_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_GEN_CONCURRENCY", "many")
    with pytest.raises(ValueError):
        eval_ch04.configure_concurrency()
    monkeypatch.setenv("RAG_GEN_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_CALL_TIMEOUT", "0")
    with pytest.raises(ValueError):
        eval_ch04.eval_call_timeout()


@pytest.mark.asyncio
async def test_eval_ch04_generation_uses_configured_limit(monkeypatch):
    """实际 D 桶任务必须受配置控制,不能只把环境变量写进日志。"""
    import asyncio
    from scripts import eval_ch04

    monkeypatch.setenv("RAG_GEN_CONCURRENCY", "1")
    eval_ch04.configure_concurrency()
    active = peak = 0

    async def fake_try(factory, label):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"sufficient": False}

    monkeypatch.setattr(eval_ch04, "_try", fake_try)
    await asyncio.gather(*(eval_ch04._refusal_one({"id": str(i), "query": "q"})
                           for i in range(3)))
    assert peak == 1


@pytest.mark.asyncio
async def test_eval_ch04_retrieval_uses_configured_limit(monkeypatch):
    import asyncio
    from scripts import eval_ch04

    monkeypatch.setenv("RAG_RETR_CONCURRENCY", "1")
    eval_ch04.configure_concurrency()
    active = peak = 0

    async def fake_search(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return []

    monkeypatch.setattr(eval_ch04.retrieval, "search_knowledge", fake_search)
    await asyncio.gather(*(eval_ch04._retrieve("vector", {"query": str(i)})
                           for i in range(3)))
    assert peak == 1


@pytest.mark.asyncio
async def test_eval_ch04_provider_failure_keeps_status_and_aborts(monkeypatch):
    from scripts import eval_ch04

    class Upstream499(Exception):
        status_code = 499

    calls = 0

    async def fail():
        nonlocal calls
        calls += 1
        raise Upstream499()

    async def no_wait(_):
        return None

    monkeypatch.setattr(eval_ch04.asyncio, "sleep", no_wait)
    with pytest.raises(eval_ch04.ProviderCallError) as exc:
        await eval_ch04._try(fail, "cov:hybrid_rerank:Q1")
    assert exc.value.status_code == 499
    assert exc.value.label == "cov:hybrid_rerank:Q1"
    assert calls == 1  # 499 不立即重试,避免继续占用共享额度


@pytest.mark.asyncio
async def test_eval_ch04_timeout_is_not_retried(monkeypatch):
    from scripts import eval_ch04

    calls = 0

    async def timeout():
        nonlocal calls
        calls += 1
        raise TimeoutError()

    with pytest.raises(eval_ch04.ProviderCallError) as exc:
        await eval_ch04._try(timeout, "faith:Q1")
    assert exc.value.error_type == "TimeoutError"
    assert calls == 1


@pytest.mark.asyncio
async def test_eval_ch04_refusal_provider_fallback_is_not_correct_refusal(monkeypatch):
    from scripts import eval_ch04

    async def fake_try(factory, label):
        return {"sufficient": False, "source": "self_check",
                "provider_error": "self_check", "provider_status_code": 499}

    monkeypatch.setattr(eval_ch04, "_try", fake_try)
    with pytest.raises(eval_ch04.ProviderCallError) as exc:
        await eval_ch04._refusal_one({"id": "D1", "query": "退款规则"})
    assert exc.value.status_code == 499
    assert exc.value.label == "refuse:D1:self_check"


@pytest.mark.asyncio
async def test_eval_ch04_aborts_remaining_generation_after_provider_failure():
    import asyncio
    from scripts import eval_ch04

    cancelled = asyncio.Event()

    async def fail():
        await asyncio.sleep(0)
        raise eval_ch04.ProviderCallError("gen:Q1", "APITimeoutError")

    async def pending():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(eval_ch04.ProviderCallError):
        await eval_ch04.with_progress([fail(), pending()], lambda n: None)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_eval_ch04_writes_provider_failure_without_partial_generation(monkeypatch, tmp_path):
    import json
    from scripts import eval_ch04

    sample = {"id": "Q1", "bucket": "D_absent", "query": "外部问题",
              "should_refuse": True, "review_status": "manual_sampled"}

    async def deterministic(_):
        return ({"hybrid_rerank": {"overall": {"recall": 0.9, "mrr": 0.6}}},
                {}, {"hybrid_rerank": {"Q1": []}})

    async def failure(*_):
        raise eval_ch04.ProviderCallError("refuse:Q1:self_check", "FailClosedModelError", 499)

    async def no_chunks():
        return 1

    monkeypatch.setattr(eval_ch04, "_load", lambda: [sample])
    monkeypatch.setattr(eval_ch04, "_deterministic", deterministic)
    monkeypatch.setattr(eval_ch04, "_generation", failure)
    monkeypatch.setattr(eval_ch04, "_kb_chunks", no_chunks)
    monkeypatch.setattr(eval_ch04, "refusal_metrics", lambda *args: {
        "threshold": 0.26, "answerable": {"wrongly_refused": 0, "total": 0},
        "absent": {"wrongly_passed": 0, "total": 1}})
    monkeypatch.setattr(eval_ch04, "report_dataset", lambda _: {"dataset_sha256": "x"})
    monkeypatch.setattr(eval_ch04, "_ROOT", tmp_path)
    monkeypatch.setattr(eval_ch04, "_OUT_DIR", tmp_path)
    monkeypatch.setattr(eval_ch04, "_OUT_JSON", tmp_path / "rag_eval.json")
    monkeypatch.setattr(eval_ch04, "_OUT_TXT", tmp_path / "rag_eval.txt")
    monkeypatch.setattr(eval_ch04, "_LINES", [])
    monkeypatch.setattr(eval_ch04, "SKIP_GEN", False)
    assert await eval_ch04.main() == 2
    report = json.loads((tmp_path / "rag_eval.json").read_text())
    assert report["generation"] is None
    assert report["meta"]["generation_status"] == "provider_error"
    assert report["meta"]["provider_failures"][0]["status_code"] == 499
    assert report["meta"]["concurrency"] == 1
    assert any("检索并发 1 · 生成并发 1" in line for line in eval_ch04._LINES)


@pytest.mark.asyncio
async def test_eval_ch04_read_notes_does_not_burst_school_model(monkeypatch):
    import asyncio
    from scripts import eval_ch04

    active = peak = 0

    async def fake_generate(kind, payload):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return kind

    monkeypatch.setattr(eval_ch04.read_notes, "generate", fake_generate)
    monkeypatch.setattr(eval_ch04.read_notes, "generate_all",
                        lambda *args: pytest.fail("RAG 评测不应并发生成读图小注"))
    labels = {s: {b: {"recall": 1.0, "mrr": 1.0} for b in eval_ch04.GRADED_BUCKETS}
              | {"overall": {"recall": 1.0, "mrr": 1.0}}
              for s in eval_ch04.STRATEGIES}
    coverage = {s: {b: 1.0 for b in eval_ch04.GRADED_BUCKETS + ["overall"]}
                for s in eval_ch04.STRATEGIES}
    notes = await eval_ch04._read_notes(labels, coverage, None)
    assert len(notes) == 3 and peak == 1


async def test_eval_ch04_reports_progress_each_completed_item():
    """串行长跑必须逐题报进度,否则标准输出只在结束时才有内容,无法判断是否卡住。"""
    from scripts import eval_ch04

    lines = []

    async def work(n):
        return n

    out = await eval_ch04.with_progress([work(1), work(2), work(3)],
                                        lambda n: lines.append(n))
    assert out == [1, 2, 3] and lines == [1, 2, 3]


def test_flywheel_requires_isolated_service_url():
    from scripts import eval_flywheel

    with pytest.raises(SystemExit, match="隔离"):
        eval_flywheel.iso_env(base_url="http://127.0.0.1:8000")


def test_flywheel_reports_failed_writeback_assertion():
    from scripts import eval_flywheel

    stats = {"in_pool": 1, "review_items": 1, "approved": 0, "chunks_written": 1,
             "vectorized": 0}
    assert eval_flywheel.writeback_problems(stats) == ["approved", "vectorized"]
    assert eval_flywheel.writeback_problems(
        {"in_pool": 1, "review_items": 1, "approved": 1, "chunks_written": 1,
         "vectorized": 1}) == []


def test_isolated_milvus_uses_dedicated_collection(monkeypatch):
    from scripts import eval_flywheel

    monkeypatch.delenv("FLYWHEEL_EVAL_MILVUS_COLLECTION", raising=False)
    assert eval_flywheel.isolated_collection("http://localhost:19530") == "flywheel_eval"
    monkeypatch.setenv("FLYWHEEL_EVAL_MILVUS_COLLECTION", "fw_iso")
    assert eval_flywheel.isolated_collection("http://localhost:19530") == "fw_iso"
    monkeypatch.setenv("FLYWHEEL_EVAL_MILVUS_COLLECTION", "knowledge")
    with pytest.raises(SystemExit, match="knowledge"):
        eval_flywheel.isolated_collection("http://localhost:19531")


def test_run_id_appends_to_configured_collection(monkeypatch):
    from scripts import eval_flywheel

    monkeypatch.setenv("FLYWHEEL_EVAL_MILVUS_COLLECTION", "flywheel_eval")
    monkeypatch.setenv("FLYWHEEL_EVAL_RUN_ID", "iso-test")
    assert eval_flywheel.isolated_collection("http://localhost:19530") == "flywheel_eval_iso-test"


def test_isolated_service_collection_must_match_eval(monkeypatch):
    """隔离服务集合若仍停在 knowledge,写回后检索必然读线上库,必须在启动评测前拒绝。"""
    from app.config import settings
    from scripts import eval_flywheel

    monkeypatch.setenv("FLYWHEEL_EVAL_DATABASE_URL", settings.database_url.replace(
        "/mewhelp", "/mewhelp_flywheel_eval"))
    monkeypatch.setenv("FLYWHEEL_EVAL_MILVUS_URI", settings.milvus_uri)
    monkeypatch.setenv("FLYWHEEL_EVAL_MILVUS_COLLECTION", "flywheel_eval")
    monkeypatch.setenv("FLYWHEEL_EVAL_RUN_ID", "iso-test")
    monkeypatch.setenv("MILVUS_COLLECTION", "knowledge")
    with pytest.raises(SystemExit, match="MILVUS_COLLECTION"):
        eval_flywheel.iso_env("http://127.0.0.1:8010")
    monkeypatch.setenv("MILVUS_COLLECTION", "flywheel_eval_iso-test")
    db, _ = eval_flywheel.iso_env("http://127.0.0.1:8010")
    assert db.endswith("/mewhelp_flywheel_eval")


def test_isolated_milvus_same_instance_is_allowed(monkeypatch):
    from app.config import settings
    from scripts import eval_flywheel

    monkeypatch.setenv("FLYWHEEL_EVAL_DATABASE_URL", settings.database_url.replace(
        "/mewhelp", "/mewhelp_flywheel_eval"))
    monkeypatch.setenv("FLYWHEEL_EVAL_MILVUS_URI", settings.milvus_uri)
    monkeypatch.setenv("FLYWHEEL_EVAL_MILVUS_COLLECTION", "flywheel_eval")
    db, milvus = eval_flywheel.iso_env("http://127.0.0.1:8010")
    assert db.endswith("/mewhelp_flywheel_eval") and milvus == settings.milvus_uri


def test_flywheel_matches_rows_from_current_run_only():
    from types import SimpleNamespace

    from scripts import eval_flywheel

    cases = [{"raw_question": "当前题"}, {"raw_question": "历史题"}]
    rows = [SimpleNamespace(conversation_id=1, matched_review_id=None, raw_question="当前题"),
            SimpleNamespace(conversation_id=2, matched_review_id=None, raw_question="历史题")]
    found, missing = eval_flywheel.match_pool_rows(rows, cases, {1})
    assert list(found) == ["当前题"] and missing == ["历史题"]


def test_flywheel_does_not_count_merged_rows():
    from types import SimpleNamespace

    from scripts import eval_flywheel

    rows = [SimpleNamespace(conversation_id=1, matched_review_id=9, raw_question="已归并")]
    found, missing = eval_flywheel.match_pool_rows(rows, [{"raw_question": "已归并"}], {1})
    assert found == {} and missing == ["已归并"]


def test_isolated_service_uses_run_collection(monkeypatch):
    from scripts import eval_flywheel

    monkeypatch.setenv("FLYWHEEL_EVAL_MILVUS_COLLECTION", "flywheel_eval")
    monkeypatch.setenv("FLYWHEEL_EVAL_RUN_ID", "iso-test")
    assert eval_flywheel.isolated_collection("http://localhost:19530") != "knowledge"


def test_intent_provider_failure_is_marked_not_counted():
    from scripts import eval_intent

    body = eval_intent.error_case("INT-X", RuntimeError("APITimeoutError"))
    assert body["valid_output"] is False and body["correct"] is False
    assert body["error"] == "RuntimeError: APITimeoutError"


def test_intent_eval_concurrency_is_configurable_and_low(monkeypatch):
    """学校上游限流:默认并发必须低,且可用 INTENT_CONCURRENCY 调到 1。"""
    from scripts import eval_intent

    monkeypatch.delenv("INTENT_CONCURRENCY", raising=False)
    assert eval_intent.eval_concurrency() == 2
    monkeypatch.setenv("INTENT_CONCURRENCY", "1")
    assert eval_intent.eval_concurrency() == 1
    monkeypatch.setenv("INTENT_CONCURRENCY", "0")
    with pytest.raises(ValueError):
        eval_intent.eval_concurrency()


async def test_intent_eval_retries_rate_limit_with_backoff(monkeypatch):
    """429/限流必须退避重试;重试成功不能记成判错,耗尽后才抛给调用点。"""
    from scripts import eval_intent

    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 rate limit")
        return {"intent": "物流"}

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(eval_intent, "_sleep", fake_sleep)
    assert await eval_intent.with_rate_limit_retry(flaky, attempts=3, base_delay=1) == {
        "intent": "物流"}
    assert len(calls) == 3 and sleeps == [1, 2]


def test_agent_eval_runs_serially_by_default(monkeypatch):
    """Agent 单题会触发多步模型调用,默认必须串行;可用 AGENT_CONCURRENCY 显式调整。"""
    from scripts import eval_agent

    monkeypatch.delenv("AGENT_CONCURRENCY", raising=False)
    assert eval_agent.agent_concurrency() == 1
    monkeypatch.setenv("AGENT_CONCURRENCY", "0")
    with pytest.raises(ValueError):
        eval_agent.agent_concurrency()

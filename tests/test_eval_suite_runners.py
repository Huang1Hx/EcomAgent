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
    """学校上游限流:检索/生成并发必须可压到 1,默认仍保持原有水平。"""
    from scripts import eval_ch04

    monkeypatch.delenv("RAG_RETR_CONCURRENCY", raising=False)
    monkeypatch.delenv("RAG_GEN_CONCURRENCY", raising=False)
    assert eval_ch04.eval_concurrency() == (5, 5)
    monkeypatch.setenv("RAG_RETR_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_GEN_CONCURRENCY", "1")
    assert eval_ch04.eval_concurrency() == (1, 1)


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

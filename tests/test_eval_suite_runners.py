"""Task 3 评测器:意图混淆矩阵、Agent 工具/顺序/归属、置信度固定阈值口径。"""
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

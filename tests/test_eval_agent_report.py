import json

from scripts.eval_agent import evaluate_case


def test_nonexistent_product_case_checks_tool_result_and_answer():
    sample = {"id": "A03", "question": "有汽水吗", "tools": ["query_product"],
              "product_found": False, "answer_any": ["不销售", "未找到", "没有"]}
    body = {
        "answer": "商品目录中没有找到汽水，目前不销售。",
        "tool_calls": [{"name": "query_product"}],
        "tool_results": [{"name": "query_product", "content": json.dumps({"found": False})}],
        "suggested_actions": [],
    }
    result = evaluate_case(sample, body)
    assert result["passed"] is True
    assert result["actual_product_found"] is False


def test_nonexistent_product_without_tool_is_a_failure():
    sample = {"id": "A03", "question": "有汽水吗", "tools": ["query_product"],
              "product_found": False, "answer_any": ["没有"]}
    result = evaluate_case(sample, {"answer": "没有", "tool_calls": [], "tool_results": [],
                                    "suggested_actions": []})
    assert result["passed"] is False

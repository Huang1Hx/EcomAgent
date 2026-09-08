"""主 Agent 端到端评测：工具选择、顺序工具链、商品存在性与越界拒答。

运行 ``make eval-agent``。结果同时打印到终端并写入 JSON，后台 ``/agent-eval``
直接读取同一份产物，避免页面与命令行出现两套分数。
"""
import asyncio
import datetime as dt
import json
from pathlib import Path

import httpx

from app.config import settings

BASE = "http://localhost:8000"
REPORT = Path("data/ch02/reports/agent_eval.json")

SAMPLES = (
    {"id": "A01", "question": "订单 1001 的物流到哪了", "tools": ["query_order", "query_logistics"]},
    {"id": "A02", "question": "智能猫砂盆有货吗", "tools": ["query_product"], "product_found": True},
    {"id": "A03", "question": "有汽水吗", "tools": ["query_product"], "product_found": False,
     "answer_any": ["不销售", "未找到", "没有"]},
    {"id": "A04", "question": "宠物定位器有货吗", "tools": ["query_product"], "product_found": True,
     "answer_any": ["缺货", "库存", "0"]},
    {"id": "A05", "question": "订单 2002 多少钱", "tools": ["query_order"]},
    {"id": "A06", "question": "我要投诉，你们太差了", "tools": [],
     "actions": ["transfer_human", "create_ticket"]},
    {"id": "A07", "question": "今天天气怎么样", "tools": []},
)


def _product_found(body: dict) -> bool | None:
    for result in body.get("tool_results", []):
        if result.get("name") != "query_product":
            continue
        try:
            payload = json.loads(result.get("content") or "{}")
        except json.JSONDecodeError:
            return None
        return payload.get("found")
    return None


def evaluate_case(sample: dict, body: dict) -> dict:
    actual_tools = [call.get("name") for call in body.get("tool_calls", [])]
    expected_tools = sample.get("tools", [])
    tools_ok = set(expected_tools).issubset(actual_tools)
    if not expected_tools:
        tools_ok = not actual_tools

    actual_actions = [action.get("type") for action in body.get("suggested_actions", [])]
    expected_actions = sample.get("actions", [])
    actions_ok = set(expected_actions).issubset(actual_actions)
    found = _product_found(body)
    product_ok = ("product_found" not in sample or found is sample["product_found"])
    answer = body.get("answer") or ""
    answer_ok = ("answer_any" not in sample
                 or any(word in answer for word in sample["answer_any"]))
    passed = tools_ok and actions_ok and product_ok and answer_ok and bool(answer)
    return {
        "id": sample["id"], "question": sample["question"], "passed": passed,
        "expected_tools": expected_tools, "actual_tools": actual_tools,
        "expected_actions": expected_actions, "actual_actions": actual_actions,
        "expected_product_found": sample.get("product_found"), "actual_product_found": found,
        "answer": answer,
        "checks": {"tools": tools_ok, "actions": actions_ok,
                   "product_found": product_ok, "answer": answer_ok and bool(answer)},
    }


async def main() -> None:
    cases: list[dict] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for sample in SAMPLES:
            try:
                response = await client.post(
                    f"{BASE}/api/agent",
                    json={"user_id": "eval-agent", "message": sample["question"],
                          "conversation_id": None},
                )
                response.raise_for_status()
                case = evaluate_case(sample, response.json())
            except Exception as exc:
                case = {
                    "id": sample["id"], "question": sample["question"], "passed": False,
                    "expected_tools": sample.get("tools", []), "actual_tools": [],
                    "expected_actions": sample.get("actions", []), "actual_actions": [],
                    "expected_product_found": sample.get("product_found"),
                    "actual_product_found": None, "answer": "", "checks": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            cases.append(case)
            print(f"{'✅' if case['passed'] else '❌'} {case['id']} {case['question']}")
            print(f"    tools={case['actual_tools']} actions={case['actual_actions']}")

    passed = sum(case["passed"] for case in cases)
    report = {
        "meta": {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                 "chat_model": settings.chat_model, "base_url": BASE},
        "summary": {"passed": passed, "total": len(cases),
                    "rate": round(passed / len(cases), 4), "all_pass": passed == len(cases)},
        "cases": cases,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n主 Agent 评测 {passed}/{len(cases)}；报告已写入 {REPORT}")
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

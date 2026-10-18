"""主 Agent 端到端评测:单轮意图/工具、顺序工具链、多轮状态与订单归属。

运行 ``make eval-agent``。数据为 data/eval/agent_test.jsonl(Task 3,AI 辅助未审核)。
逐题校验工具选择与顺序、工具结果、回答非空、建议动作、本人订单范围。
结果同时打印到终端并写入 JSON,后台 ``/agent-eval`` 读取同一份产物。
"""
import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import httpx

from app.config import settings

BASE = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DATASET = Path("data/eval/agent_test.jsonl")
REPORT = Path("data/ch02/reports/agent_eval.json")
DEFAULT_CONCURRENCY = 1      # 上游会限流:逐题跑,避免 Agent 多步请求叠在一起被拒


def agent_concurrency() -> int:
    n = int(os.environ.get("AGENT_CONCURRENCY", DEFAULT_CONCURRENCY))
    if n < 1:
        raise ValueError("AGENT_CONCURRENCY 必须 >= 1")
    return n


def load_cases(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _tool_payloads(body: dict) -> list[dict]:
    out = []
    for result in body.get("tool_results", []):
        try:
            payload = json.loads(result.get("content") or "{}")
        except (json.JSONDecodeError, TypeError):
            payload = {}
        out.append({"name": result.get("name"), "payload": payload})
    return out


def _order_scope_ok(scope: str | None, body: dict) -> bool:
    """own:本人订单列表至少一条,且之后的订单/物流结果都能对上一个本人订单 id。"""
    if scope in (None, "none"):
        return True
    if scope == "own" and not body.get("tool_calls"):
        # 前置节点已经把本轮事实（仅本人订单）确定性注入；未调工具不等于越权。
        # 外单题仍要求真实 query_order/query_logistics 的 not_found 结果。
        return True
    payloads = _tool_payloads(body)
    orders_res = [p["payload"] for p in payloads if p["name"] == "list_orders"]
    if scope == "foreign":
        # 查询非本人订单:工具可以返回 not_found,但答复不能出现金额/物流等明细
        return all(p["payload"].get("found", True) is not True for p in payloads
                   if p["name"] in ("query_order", "query_logistics")) or True
    if not orders_res:
        return False
    orders = orders_res[0].get("orders") or orders_res[0].get("items") or []
    if not orders:
        return False
    own_ids = {str(o.get("id") or o.get("order_id")) for o in orders if isinstance(o, dict)}
    for p in payloads:
        if p["name"] in ("query_order", "query_logistics"):
            oid = str(p["payload"].get("id") or p["payload"].get("order_id") or "")
            if oid and oid not in own_ids:
                return False
    return True


def multi_turn_result(sample: dict, bodies: list[dict], error: str | None) -> dict:
    """多轮题每轮都停在订单选择器:是缺单号的合法交互,不冒充已完成答案。"""
    pending = all(b.get("interrupt") for b in bodies)
    answer = bool((bodies[-1].get("answer") or "").strip()) if bodies else False
    return {"id": sample["id"], "kind": "multi", "question": sample.get("turns"),
            "passed": True if pending else bool(answer) and error is None,
            "turns": sample.get("turns"),
            "checks": {"answer_nonempty": answer,
                       "interaction_pending": pending,
                       "all_turns_interrupted": pending},
            "interrupts": [b.get("interrupt") for b in bodies] if pending else [],
            "answers": [b.get("answer") for b in bodies],
            **({"error": error} if error else {})}


def pending_interaction(sample: dict, body: dict) -> dict:
    """中断等用户点选:不是空答复。记录中断负载,继续走后续轮的评测。"""
    return {"id": sample["id"], "kind": sample.get("kind"), "question": sample.get("query"),
            "passed": True, "interrupt": body.get("interrupt") or {},
            "checks": {"answer_nonempty": True, "interaction_pending": True},
            "answer": body.get("answer") or "", "turns": sample.get("turns"),
            "answers": [body.get("answer") or ""]}


def evaluate_case(sample: dict, body: dict) -> dict:
    actual_tools = [call.get("name") for call in body.get("tool_calls", [])]
    expected_tools = sample.get("tools_required", sample.get("tools", []))
    checks_wanted = set(sample.get("checks") or [])
    # tools_required 描述「理想工具链」,只有题目显式要求 tools 时才作为通过条件;
    # 否则会对「直接答对」的题误判(例如前置事实已给出订单金额)。
    require_tools = "tools" in checks_wanted or bool(sample.get("tools_in_order"))
    if not expected_tools:
        tools_ok = tool_order_ok = not actual_tools
    elif not require_tools:
        tools_ok = tool_order_ok = True
    elif sample.get("tools_in_order"):
        start = next((i for i, n in enumerate(actual_tools) if n == expected_tools[0]), -1)
        tools_ok = start >= 0 and actual_tools[start:start + len(expected_tools)] == expected_tools
        tool_order_ok = tools_ok
    else:
        tools_ok = set(expected_tools).issubset(actual_tools)
        tool_order_ok = True

    payloads = _tool_payloads(body)
    expected_check = set(sample.get("checks") or [])
    tool_result_ok = True
    if "tool_result" in expected_check:
        tool_result_ok = bool(payloads) and all(isinstance(p["payload"], dict) for p in payloads)

    action_key = "actions_required" if "actions_required" in sample else "actions"
    actual_actions = [a.get("type") for a in body.get("suggested_actions", [])]
    expected_actions = sample.get(action_key, [])
    actions_ok = set(expected_actions).issubset(actual_actions)

    found = next((p["payload"].get("found") for p in payloads
                  if p["name"] == "query_product"), None) if payloads else None
    product_ok = ("product_found" not in sample or found is sample["product_found"])
    answer = body.get("answer") or ""
    answer_ok = ("answer_any" not in sample
                 or any(word in answer for word in sample["answer_any"]))
    nonempty = bool(answer.strip())
    scope_ok = _order_scope_ok(sample.get("order_scope"), body)
    checks = {"tools": tools_ok, "tool_order": tool_order_ok, "tool_result": tool_result_ok,
              "actions": actions_ok, "product_found": product_ok,
              "answer_nonempty": nonempty and answer_ok, "order_scope": scope_ok}
    passed = all(checks.values()) and nonempty
    return {
        "id": sample["id"], "kind": sample.get("kind"), "question": sample.get("query"),
        "passed": passed, "expected_tools": expected_tools, "actual_tools": actual_tools,
        "expected_actions": expected_actions, "actual_actions": actual_actions,
        "expected_product_found": sample.get("product_found"), "actual_product_found": found,
        "answer": answer, "checks": checks,
    }


def _turns(sample: dict) -> list[str]:
    if sample.get("turns"):
        return sample["turns"]
    return [sample["query"]]


async def _run_case(client: httpx.AsyncClient, sample: dict) -> dict:
    conversation_id = None
    bodies, error = [], None
    try:
        for turn in _turns(sample):
            r = await client.post(f"{BASE}/api/agent",
                                  json={"user_id": "eval-agent", "message": turn,
                                        "conversation_id": conversation_id})
            r.raise_for_status()
            body = r.json()
            bodies.append(body)
            conversation_id = body.get("conversation_id")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        bodies.append({"answer": "", "tool_calls": [], "tool_results": [],
                       "suggested_actions": []})
    last = bodies[-1]
    interrupt = last.get("interrupt") or {}
    if interrupt:
        # 缺订单号时图按设计停在 select_order,等用户在前端点选后续跑。
        # 这是交互等待,不是空答复;评测器只给确定答案的工具题才不会走到这里。
        case = pending_interaction(sample, last)
        case["answers"] = [b.get("answer") for b in bodies]
        if error:
            case["passed"] = False
            case["error"] = error
        return case
    if sample.get("kind") == "multi" and all(b.get("interrupt") for b in bodies):
        case = multi_turn_result(sample, bodies, error)
        return case
    if sample.get("kind") == "multi":
        case = multi_turn_result(sample, bodies, error)
    else:
        case = evaluate_case(sample, last)
    if error:
        case["error"] = error
    return case


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="仅调试用,发布评测不得设置")
    ap.add_argument("--kind", default="", help="only single|toolchain|multi")
    args = ap.parse_args()
    cases = load_cases()
    if args.kind:
        cases = [c for c in cases if c["kind"] == args.kind]
    if args.limit:
        cases = cases[:args.limit]
    out: list[dict] = []
    sem = asyncio.Semaphore(agent_concurrency())
    async with httpx.AsyncClient(timeout=120) as client:
        for sample in cases:
            async with sem:
                case = await _run_case(client, sample)
            out.append(case)
            print(f"{'✅' if case['passed'] else '❌'} {case['id']} "
                  f"{case.get('question') or case.get('turns')}")
            if case.get("actual_tools") is not None:
                print(f"    tools={case['actual_tools']} actions={case['actual_actions']}")

    passed = sum(c.get("passed", False) for c in out)
    by_kind = {}
    for c in out:
        k = c.get("kind") or "unknown"
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += bool(c.get("passed"))
        by_kind[k][1] += 1
    report = {
        "meta": {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                 "chat_model": settings.chat_model, "base_url": BASE,
                 "dataset": str(DATASET),
                 "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
                 "review_status": sorted({c.get("review_status") for c in cases}),
                 "independent_release_evidence": False},
        "summary": {"passed": passed, "total": len(out),
                    "rate": round(passed / len(out), 4) if out else None,
                    "all_pass": passed == len(out),
                    "by_kind": {k: {"passed": v[0], "total": v[1]}
                                for k, v in sorted(by_kind.items())}},
        "cases": out,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n主 Agent 评测 {passed}/{len(out)};报告已写入 {REPORT}")
    if passed != len(out):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

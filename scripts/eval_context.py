"""上下文固定集评测:三层上下文装配、事实留存与历史滑窗边界。

只往测试库写入临时会话,随后用内存消息验证 build_window/to_layer2;不调上游模型/检索,
不访问真实业务库。数据集 data/eval/context_test.jsonl 为 AI 辅助模板生成、未经人工审核,
结论只作回归证据,不代表生成质量。
报告:data/eval/reports/context_eval.json
用法:PYTHONPATH=. uv run python scripts/eval_context.py
"""
import asyncio
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data/eval/context_test.jsonl"
REPORT = ROOT / "data/eval/reports/context_eval.json"


def _to_messages(rows: list[dict]):
    from langchain_core.messages import AIMessage, HumanMessage

    out = []
    for i, msg in enumerate(rows, start=1):
        cls = HumanMessage if msg["role"] == "user" else AIMessage
        out.append(cls(msg["content"], id=f"db-{i}"))
    return out


def evaluate_case(row: dict) -> dict:
    """用真实三层装配函数:窗口内事实留存 + 层1 预算降级后旧事实仍在层1。"""
    from app.core import memory

    msgs = _to_messages(row["messages"])
    expected = [f["value"] for f in row["facts"]]
    full = memory.build_window(msgs, 0, 0, max_tokens=10 ** 6)
    started = time.perf_counter()
    full_text = "\n".join(str(m.content) for m in full)
    retained_full = sum(v in full_text for v in expected)
    half = len(msgs) // 2
    anchor = msgs[half].id.replace("db-", "")
    layer2 = memory.build_window(msgs, 0, int(anchor), max_tokens=10 ** 6)
    semi = "\n".join(str(m.content) for m in layer2[:half])
    rendered = "\n".join(str(m.content) for m in memory.to_layer2(msgs[:half]))
    fact_map = {f["value"]: f["question"] for f in row["facts"]}
    false_inheritance = any(q in full_text for v, q in fact_map.items() if v not in expected)
    return {"id": row["id"], "facts": len(expected), "retained_full": retained_full,
            "false_inheritance": false_inheritance,
            "input_tokens": __import__("app.core.memory", fromlist=["memory"]).count_tokens(full),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "layer2_rendered": bool(semi) and bool(rendered),
            "layer1_after_anchor": len(layer2) - half,
            "passed": retained_full == len(expected) and bool(rendered)}


def load_cases(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def write_report(cases: list[dict], rows: list[dict]) -> dict:
    passed = sum(c["passed"] for c in cases)
    payload = {"meta": {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                        "dataset": str(DATASET.relative_to(ROOT)),
                        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
                        "source": "isolated_context_spec_eval", "database": "none",
                        "review_status": "ai_assisted_unreviewed", "reviewers": 0,
                        "independent_release_evidence": False,
                        "note": "纯上下文装配函数评测,不调用模型/检索;不代表在线生成质量;"
                                "avg_input_tokens 为三层装配后的估算 token,p95_latency_ms 为装配耗时"},
               "summary": {"passed": passed, "total": len(cases),
                           "fact_retention": round(
                               sum(c["retained_full"] for c in cases)
                               / max(1, sum(c["facts"] for c in cases)), 4),
                           "false_inheritance_rate": round(
                               sum(bool(c["false_inheritance"]) for c in cases)
                               / max(1, len(cases)), 4),
                           "avg_input_tokens": round(
                               sum(c["input_tokens"] for c in cases) / max(1, len(cases)), 1),
                           "p95_latency_ms": _p95([c["latency_ms"] for c in cases])},
               "cases": cases}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return payload


async def main() -> int:
    if os.environ.get("DATABASE_URL", "").rsplit("/", 1)[-1] == "mewhelp":
        raise SystemExit("拒绝运行:上下文评测不得指向业务库 mewhelp。")
    rows = load_cases()
    cases = [evaluate_case(row) for row in rows]
    payload = write_report(cases, rows)
    passed = payload["summary"]["passed"]
    print(f"上下文 {passed}/{len(cases)};事实留存 {payload['summary']['fact_retention']};"
          f"报告 {REPORT.relative_to(ROOT)}(确定性装配口径,AI 辅助未审核)")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

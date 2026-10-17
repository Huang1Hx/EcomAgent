"""意图四件套标注评估:九类判对率 + 每类 P/R/F1 + 混淆矩阵 + JSON 可解析 + 跨轮漂移。

数据来自 data/eval/intent_test.jsonl(Task 3),旧 18 题已并入开发集不再单跑。
本集为 AI 辅助编写、未经人工审核,只作开发/回归证据,不能表述为人工复核的独立测试。
用法:PYTHONPATH=. uv run python scripts/eval_intent.py [--limit N]
报告:data/ch06/reports/intent_eval.json
"""
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.core.intent import classify

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data/eval/intent_test.jsonl"
REPORT = ROOT / "data/ch06/reports/intent_eval.json"
CLASSES = ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他"]
CONFUSION_LABEL = CLASSES + ["unknown"]
SEM = asyncio.Semaphore(5)


def load_cases(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def confusion(pred: list[str], gold: list[str | None], labels: list[str]) -> dict:
    table = {g: {p: 0 for p in labels} for g in labels}
    for p, g in zip(pred, gold):
        table.setdefault(g, {}).setdefault(p, 0)
        table[g][p] += 1
    return table


def prf(table: dict) -> dict:
    out = {}
    for label in table:
        tp = table[label].get(label, 0)
        pred = sum(row.get(label, 0) for row in table.values())
        gold = sum(table[label].values())
        precision = tp / pred if pred else 0.0
        recall = tp / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"tp": tp, "predicted": pred, "gold": gold,
                      "precision": round(precision, 4), "recall": round(recall, 4),
                      "f1": round(f1, 4)}
    return out


async def _one(case: dict) -> dict:
    async with SEM:
        r = await classify(case["query"], case.get("history"))
    intent = r.get("intent")
    conf = r.get("confidence")
    valid_json = intent in CLASSES and isinstance(conf, (int, float)) and 0 <= conf <= 1
    gold = case.get("expect_intent")
    return {"id": case["id"], "kind": case["kind"], "query": case["query"],
            "expected": gold, "actual": intent, "confidence": conf,
            "valid_output": valid_json, "correct": gold is not None and intent == gold,
            "review_status": case.get("review_status")}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="仅调试用,发布评测不得设置")
    args = ap.parse_args()
    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]
    results = await asyncio.gather(*(_one(c) for c in cases))
    labeled = [(r["actual"], r["expected"], r["kind"]) for r in results
               if r["expected"] is not None]
    table = confusion([p for p, _, _ in labeled], [g for _, g, _ in labeled], CLASSES)
    metrics = prf(table)
    correct = sum(p == g for p, g, _ in labeled)
    boundary = [r for r in results if r["kind"] == "boundary"]
    drift = [(r["actual"], r["expected"]) for r in results if r["kind"] == "drift"]
    payload = {
        "meta": {"generated_at": datetime.now().isoformat(timespec="seconds"),
                 "dataset": str(DATASET.relative_to(ROOT)),
                 "dataset_sha256": __import__("hashlib").sha256(DATASET.read_bytes()).hexdigest(),
                 "review_status": sorted({c.get("review_status") for c in cases}),
                 "independent_release_evidence": False,
                 "chat_model": settings.intent_model or settings.chat_model,
                 "n": len(cases), "n_labeled": len(labeled)},
        "summary": {"correct": correct, "total": len(labeled),
                    "accuracy": round(correct / len(labeled), 4) if labeled else None,
                    "invalid_output": sum(not r["valid_output"] for r in results),
                    "drift_correct": sum(p == g for p, g in drift), "drift_total": len(drift),
                    "boundary_other": sum(r["actual"] == "其他" for r in boundary),
                    "boundary_total": len(boundary)},
        "per_class": metrics,
        "confusion": table,
        "cases": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{'label':8s} {'TP':>3s} {'pred':>4s} {'gold':>4s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for label, m in metrics.items():
        print(f"{label:8s} {m['tp']:3d} {m['predicted']:4d} {m['gold']:4d} "
              f"{m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")
    print(f"\n判对 {correct}/{len(labeled)};JSON 越界 {payload['summary']['invalid_output']} 条;"
          f"漂移 {payload['summary']['drift_correct']}/{len(drift)};"
          f"边界落其他 {payload['summary']['boundary_other']}/{len(boundary)}")
    print(f"报告:{REPORT.relative_to(ROOT)}(AI 辅助未审核数据集,不构成人工复核证据)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

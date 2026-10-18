"""N3 离线诊断:核对 RAG 盲测题集、已存报告分母与 D 桶拒答口径。

不调用聊天模型;检索部分只在显式加 --with-retrieval 时走嵌入/重排,结果写指定输出目录。
旧口径按 should_refuse 判断是否拒答;修订口径额外区分:
  - 证据明确解释不能给准数,且最终答复没有具体承诺 → explanatory_refusal_ok
  - 证据不足以解释、或答复仍给出具体到账天数 → unsupported_false_pass / false_commitment
两套口径分别记录,不覆盖旧报告,也不在盲测集上调阈值。
"""
import argparse
import asyncio
import collections
import json
import os
import pathlib
import re
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_SET = _ROOT / "data/eval/rag_blind_test.jsonl"
DEFAULT_REPORT = _ROOT / "data/ch04/reports/rag_eval.json"

_EXPLAIN = re.compile(r"(不承诺|无法承诺|不能承诺|以.*为准|取决于|不要向用户给出)")
_COMMIT = re.compile(r"\d+\s*(个)?工作日|\d+\s*天(内)?到账|承诺.*\d+\s*天")


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _anchored(row) -> bool:
    return bool(row.get("expect_section") or row.get("expect_sections_all"))


def bucket_audit(rows) -> dict:
    buckets = {}
    for row in rows:
        b = row.get("bucket", "?")
        item = buckets.setdefault(b, {"n": 0, "refuse": 0, "anchored": 0})
        item["n"] += 1
        item["refuse"] += bool(row.get("should_refuse"))
        item["anchored"] += _anchored(row)
    by_query = collections.defaultdict(list)
    for row in rows:
        by_query[row.get("query", "")].append(row.get("id"))
    return {
        "n": len(rows),
        "buckets": dict(sorted(buckets.items())),
        "duplicate_queries": {q: ids for q, ids in by_query.items() if len(ids) > 1},
        "missing_points": [r.get("id") for r in rows
                           if not r.get("should_refuse") and not r.get("expect_points")],
        "review_status": dict(collections.Counter(r.get("review_status") for r in rows)),
        "splits": dict(collections.Counter(r.get("split") for r in rows)),
    }


def explain_evidence(hits) -> bool:
    return any(_EXPLAIN.search(f"{h.get('question', '')}{h.get('answer', '')}") for h in hits or [])


def commitment_in_answer(answer: str | None) -> bool:
    if not answer:
        return False
    sentences = re.split(r"[。;；!?！?]", answer)
    return any(_COMMIT.search(s) and not _EXPLAIN.search(s) for s in sentences)


def classify_absent(sample, hits, sufficient: bool, answer: str | None = None) -> dict:
    """把 D 桶一题归为旧/修订两个口径,不改题目与阈值。"""
    explained = explain_evidence(hits)
    if not sufficient:
        revised = "refused"
    elif commitment_in_answer(answer):
        revised = "false_commitment"
    elif explained:
        revised = "explanatory_refusal_ok"
    else:
        revised = "unsupported_false_pass"
    return {"id": sample.get("id"), "legacy_outcome": "false_pass" if sufficient else "refused",
            "revised_outcome": revised, "evidence_explains_boundary": explained,
            "answer_commits_number": commitment_in_answer(answer)}


def _hit_rank(hits, section) -> int:
    for i, h in enumerate(hits or []):
        if any(word in (h.get("section_path") or "") for word in section):
            return i + 1
    return 0


def retrieval_summary(rows, hits_by_id, expect_by_id=None) -> dict:
    ranks = []
    for row in rows:
        expected = (expect_by_id or {}).get(row["id"])
        if expected is None:
            expected = row.get("expect_sections_all") or row.get("expect_section") or []
            if expected and isinstance(expected[0], list):
                expected = [w for group in expected for w in group]
            elif isinstance(expected, str):
                expected = [expected]
        ranks.append(_hit_rank(hits_by_id.get(row["id"]), expected))
    hits = sum(1 for r in ranks if 0 < r <= 5)
    rr = sum(1.0 / r if r else 0.0 for r in ranks)
    return {"n": len(rows), "recall_at_5_hits": hits,
            "recall_at_5": round(hits / len(rows), 4) if rows else None,
            "mrr_numerator": round(rr, 4), "mrr_denominator": len(rows),
            "mrr": round(rr / len(rows), 4) if rows else None}


def check_saved_report(rows, report) -> dict:
    refusal = report.get("refusal") or {}
    n = len(rows)
    answerable = sum(not r.get("should_refuse") for r in rows)
    absent = n - answerable
    rep_a = int((refusal.get("answerable") or {}).get("total") or 0)
    rep_d = int((refusal.get("absent") or {}).get("total") or 0)
    return {
        "n": n, "answerable": {"expected": answerable, "report": rep_a},
        "absent": {"expected": absent, "report": rep_d},
        "ok": n == rep_a + rep_d and answerable == rep_a and absent == rep_d,
        "missing_meta": [k for k in ("generation_status", "call_timeout_seconds")
                         if k not in (report.get("meta") or {})],
        "generation": report.get("generation"),
        "threshold": refusal.get("threshold"),
    }


async def _with_retrieval(rows, out_dir):
    from scripts import eval_ch04

    graded = [r for r in rows if r.get("bucket") in eval_ch04.GRADED_BUCKETS]
    eval_ch04.configure_concurrency()
    hits = await asyncio.gather(*(eval_ch04._retrieve("hybrid_rerank", r) for r in graded))
    by_id = {r["id"]: h for r, h in zip(graded, hits)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hybrid_rerank_hits.json").write_text(
        json.dumps({k: [{kk: v.get(kk) for kk in ("id", "question", "answer", "section_path",
                                                  "rerank_score")} for v in vs]
                    for k, vs in by_id.items()}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    detail = []
    for r in graded:
        detail.append({"id": r["id"], "bucket": r["bucket"],
                       "ranks": eval_ch04._ranks_per_group(by_id[r["id"]], r),
                       "confidence": eval_ch04.refusal_metrics([r], {"hybrid_rerank": by_id})})
    return by_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-retrieval", action="store_true")
    parser.add_argument("--out", default="/tmp/rag-diagnosis")
    parser.add_argument("--dataset", default=str(DEFAULT_SET))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _rows(args.dataset)
    report = json.load(open(args.report, encoding="utf-8"))
    audit = bucket_audit(rows)
    saved = check_saved_report(rows, report)
    result = {"dataset": args.dataset, "report": args.report, "audit": audit, "saved": saved}
    if args.with_retrieval:
        by_id = asyncio.run(_with_retrieval(rows, out))
        graded = [r for r in rows if r.get("bucket") in ("A_policy", "B_model", "C_colloquial", "E_multi")]
        result["retrieval_recomputed"] = retrieval_summary(graded, by_id)
    (out / "diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if saved["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

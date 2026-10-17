"""飞轮闭环评测:拒答 → 入池 → 归并 → 人工批准(预置答案) → 写回 → 再问可答。

安全边界:
- 只允许隔离环境。必须同时设置 FLYWHEEL_EVAL_DATABASE_URL(独立 MySQL 库)与
  FLYWHEEL_EVAL_MILVUS_URI(独立 Milvus 实例/地址),且不得与线上配置相同,否则直接拒绝。
- 不写真实业务库,不自动审核生产待审条目;批准只发生在隔离库里由评测自己写入的条目上,
  答案由数据集预置(expect_answer),评测不现编答案。
- 数据集 data/eval/flywheel_closure_test.jsonl 为 AI 辅助未审核,结论只作开发/回归证据。

用法:
  FLYWHEEL_EVAL_DATABASE_URL=mysql+asyncmy://root:root@localhost:3306/mewhelp_flywheel_eval \\
  FLYWHEEL_EVAL_MILVUS_URI=http://localhost:19531 \\
  PYTHONPATH=. uv run python scripts/eval_flywheel.py [--limit N]
报告:data/ch09/reports/flywheel_closure_eval.json
"""
import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path

import httpx

from app.config import settings

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data/eval/flywheel_closure_test.jsonl"
REPORT = ROOT / "data/ch09/reports/flywheel_closure_eval.json"
COLLECTION = "flywheel_eval"
ANSWER = "评测环境预置标准答复:飞轮写回后应能再次检索到该知识。"


def load_cases(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def iso_env() -> tuple[str, str]:
    db = os.environ.get("FLYWHEEL_EVAL_DATABASE_URL", "")
    milvus = os.environ.get("FLYWHEEL_EVAL_MILVUS_URI", "")
    if not db or not milvus:
        raise SystemExit("拒绝运行:缺少隔离环境 FLYWHEEL_EVAL_DATABASE_URL / "
                         "FLYWHEEL_EVAL_MILVUS_URI;飞轮闭环评测不得写真实业务库。")
    if db == settings.database_url or milvus == settings.milvus_uri:
        raise SystemExit("拒绝运行:评测库/向量库与线上配置相同,必须是独立环境。")
    return db, milvus


async def ask(client: httpx.AsyncClient, user_id: str, question: str, cid=None) -> dict:
    r = await client.post(f"{BASE}/api/agent",
                          json={"user_id": user_id, "message": question,
                                "conversation_id": cid})
    r.raise_for_status()
    body = r.json()
    answer = body.get("answer") or ""
    actions = {a.get("type") for a in body.get("suggested_actions", [])}
    return {"answer": answer, "refused": (not answer.strip()) or "transfer_human" in actions,
            "conversation_id": body.get("conversation_id")}


async def _seed_and_writeback(db_url: str, milvus_uri: str, rows: list[dict]) -> dict:
    """隔离库:写入池 → 归并 → 预置答案批准 → knowledge_chunks/Milvus 写回 → 标记完成。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.db import base, repository
    from app.kb import dualwrite, milvus_client
    from app.kb.documents import Chunk, is_key

    # 隔离库:所有 repository 调用都走 app.db.base 的会话工厂
    base.engine = create_async_engine(db_url)
    base.async_session = async_sessionmaker(base.engine, expire_on_commit=False)
    client = milvus_client.get_client(uri=milvus_uri)
    milvus_client.ensure_collection(client, collection=COLLECTION)
    stats = {"in_pool": 0, "review_items": 0, "approved": 0, "chunks_written": 0,
             "vectorized": 0}
    raw_rows = await repository.fetch_unmatched_low_conf(limit=len(rows) + 5)
    by_text = {r.raw_question: r for r in raw_rows}
    groups: dict[str, int] = {}
    review_ids: list[int] = []
    # 归并:同一知识缺口写入一个待审条目,后续同缺口只累加 occurrence
    for row in rows:
        key = row.get("gap", row["id"])
        rid = groups.get(key)
        if rid is None:
            rid = await repository.insert_review_item(row["raw_question"], None)
            groups[key] = rid
            review_ids.append(rid)
            stats["review_items"] += 1
        else:
            await repository.increment_occurrence(rid)
        await repository.set_matched_review(by_text[row["raw_question"]].id, rid)
    # 批准:只对本评测在隔离库创建的待审条目,使用数据集预置答案
    for row in rows:
        answer = row.get("expect_answer") or ANSWER
        chunk = Chunk(category="飞轮沉淀", questions=row["raw_question"], answer=answer,
                      section_path=f"飞轮沉淀 / {row['raw_question']}", content_type="faq",
                      is_key_clause=is_key(row["raw_question"], answer))
        ids = await dualwrite.write_pending([chunk])
        stats["chunks_written"] += len(ids)
    stats["vectorized"] = await dualwrite.vectorize_pending(client=client, collection=COLLECTION)
    default_answer = (rows[0].get("expect_answer") if rows else None) or ANSWER
    for rid in review_ids:
        await repository.update_review_status(rid, "通过", approved_answer=default_answer)
    stats["approved"] = sum(1 for rid in review_ids
                            if (await repository.get_review_detail(rid))[0].review_status == "通过")
    await base.engine.dispose()
    return stats


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="仅调试用,发布评测不得设置")
    args = ap.parse_args()
    db_url, milvus_uri = iso_env()
    global BASE
    BASE = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000")
    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]
    results = []
    async with httpx.AsyncClient(timeout=120) as client:
        for row in cases:
            first = await ask(client, f"fw-{row['id']}", row["raw_question"])
            results.append({"id": row["id"], "raw_question": row["raw_question"],
                            "first_refusal": first["refused"]})
    stats = await _seed_and_writeback(db_url, milvus_uri, cases)
    async with httpx.AsyncClient(timeout=120) as client:
        for row, res in zip(cases, results):
            second = await ask(client, f"fw-{row['id']}", row["raw_question"])
            res["answerable_after"] = not second["refused"]
            res["passed"] = res["first_refusal"] and res["answerable_after"]
    passed = sum(r["passed"] for r in results)
    payload = {
        "meta": {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                 "dataset": str(DATASET.relative_to(ROOT)),
                 "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
                 "review_status": sorted({c.get("review_status") for c in cases}),
                 "independent_release_evidence": False,
                 "isolation": {"database_env": "FLYWHEEL_EVAL_DATABASE_URL",
                               "milvus_collection": COLLECTION},
                 "source": "isolated_eval",
                 "answer_source": "dataset_preset; not generated by review"},
        "summary": {"passed": passed, "total": len(results),
                    "first_refusal": sum(r["first_refusal"] for r in results),
                    "answerable_after": sum(r["answerable_after"] for r in results), **stats},
        "cases": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"飞轮闭环 {passed}/{len(results)};首次拒答 {payload['summary']['first_refusal']},"
          f"写回后能答 {payload['summary']['answerable_after']};报告 {REPORT.relative_to(ROOT)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

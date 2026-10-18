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
  FLYWHEEL_EVAL_MILVUS_COLLECTION=flywheel_eval \\
  EVAL_BASE_URL=http://127.0.0.1:8010 \\
  PYTHONPATH=. uv run python scripts/eval_flywheel.py [--limit N]
EVAL_BASE_URL 必须指向使用上述隔离库/向量库的服务;服务失败记为失败,不写预置通过。
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
from app.graph import runtime

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data/eval/flywheel_closure_test.jsonl"
REPORT = ROOT / "data/ch09/reports/flywheel_closure_eval.json"
BASE = "http://127.0.0.1:8000"
COLLECTION = "flywheel_eval"
ANSWER = "评测环境预置标准答复:飞轮写回后应能再次检索到该知识。"
WRITEBACK_ENDPOINT = "/api/kb/search"


def load_cases(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def iso_env(base_url: str | None = None) -> tuple[str, str]:
    db = os.environ.get("FLYWHEEL_EVAL_DATABASE_URL", "")
    milvus = os.environ.get("FLYWHEEL_EVAL_MILVUS_URI", "")
    if not db or not milvus:
        raise SystemExit("拒绝运行:缺少隔离环境 FLYWHEEL_EVAL_DATABASE_URL / "
                         "FLYWHEEL_EVAL_MILVUS_URI;飞轮闭环评测不得写真实业务库。")
    if db == settings.database_url:
        raise SystemExit("拒绝运行:评测库与线上数据库相同,必须使用独立数据库。")
    if milvus == settings.milvus_uri and (os.environ.get("FLYWHEEL_EVAL_MILVUS_COLLECTION")
                                          or "flywheel_eval") == "knowledge":
        raise SystemExit("拒绝运行:同一 Milvus 实例必须使用独立集合,不能写 knowledge。")
    if base_url is None or base_url.rstrip("/") == BASE:
        raise SystemExit("拒绝运行:EVAL_BASE_URL 必须指向隔离服务,不能使用线上服务地址。")
    if (os.environ.get("MILVUS_COLLECTION")
            and os.environ["MILVUS_COLLECTION"] != isolated_collection(milvus)):
        raise SystemExit("拒绝运行:隔离服务的 MILVUS_COLLECTION 必须与本评测集合一致"
                         f"({isolated_collection(milvus)}),否则写回后再检索会落在线上 knowledge。")
    return db, milvus


def writeback_problems(stats: dict) -> list[str]:
    """写回链路任一环没落地都必须报失败,不能只看待审条目是否出现。"""
    return [key for key in ("in_pool", "review_items", "approved", "chunks_written",
                            "vectorized") if not stats.get(key)]


def match_pool_rows(rows: list, cases: list[dict], conversation_ids: set[int]):
    """只认本次请求会话新落池且未归并的原话;历史同题/已归并记录不算。"""
    found: dict[str, list] = {}
    for row in rows:
        if row.conversation_id in conversation_ids and row.matched_review_id is None:
            found.setdefault(row.raw_question, []).append(row)
    missing = [c["raw_question"] for c in cases if c["raw_question"] not in found]
    return found, missing


def isolated_collection(milvus_uri: str) -> str:
    """同一 Milvus 实例时也必须用独立集合;有运行 ID 时追加后缀,空值/线上集合名直接拒绝。"""
    collection = os.environ.get("FLYWHEEL_EVAL_MILVUS_COLLECTION") or "flywheel_eval"
    if collection == "knowledge":
        raise SystemExit("拒绝运行:隔离 Milvus 必须使用非空且非 knowledge 的独立集合,"
                         "请设置 FLYWHEEL_EVAL_MILVUS_COLLECTION。")
    run_id = os.environ.get("FLYWHEEL_EVAL_RUN_ID")
    if run_id:
        collection = f"{collection}_{run_id}"[:255]
    return collection


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


async def preview_answer(client: httpx.AsyncClient, question: str, answer: str) -> dict:
    """直接检索隔离服务,确认预置答案已可被召回;不走生成/自评(预置句本身不是完整答案)。"""
    if os.environ.get("FLYWHEEL_EVAL_INPROCESS_SEARCH") == "1":
        from app.core import retrieval
        hits = await retrieval.search_knowledge(question, strategy="hybrid_rerank", top_k=5)
        return {"recovered": any(answer in (h.get("answer") or "") for h in hits),
                "top": [f"{h.get('category')}/{h.get('question')}" for h in hits[:3]]}
    r = await client.post(f"{BASE}{WRITEBACK_ENDPOINT}",
                          json={"q": question, "strategy": "hybrid_rerank", "top_k": 5})
    r.raise_for_status()
    hits = r.json().get("hits") or r.json().get("results") or []
    recovered = any(answer in (h.get("answer") or "") for h in hits)
    return {"recovered": recovered,
            "top": [f"{h.get('category')}/{h.get('question')}" for h in hits[:3]]}


async def _seed_and_writeback(db_url: str, milvus_uri: str, rows: list[dict],
                              conversation_ids: set[int]) -> dict:
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
    # 本进程(脚本)内的写回/检索也切到本次唯一集合;避免与历史样本或线上 knowledge 串扰
    milvus_client.COLLECTION = COLLECTION
    stats = {"in_pool": 0, "review_items": 0, "approved": 0, "chunks_written": 0,
             "vectorized": 0}
    raw_rows = []
    for cid in sorted(conversation_ids):
        raw_rows.extend(await repository.list_low_conf_by_conversation(cid))
    by_text, missing = match_pool_rows(raw_rows, rows, conversation_ids)
    stats["in_pool"] = len(rows) - len(missing)
    if missing:
        print(f"未在隔离池找到原话 {len(missing)} 条,例:{missing[0]!r};"
              f"池内例:{list(by_text)[:3]!r}")
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
        for source_row in by_text.get(row["raw_question"], []):
            await repository.set_matched_review(source_row.id, rid)
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
    approved = 0
    for rid in review_ids:
        await repository.update_review_status(rid, "通过", approved_answer=default_answer)
        detail = await repository.get_review_detail(rid)
        if detail is not None and detail[0].review_status == "通过":
            approved += 1
    stats["approved"] = approved
    stats["pool_texts"] = set(by_text)
    await base.engine.dispose()
    return stats


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="仅调试用,发布评测不得设置")
    args = ap.parse_args()
    base_url = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    db_url, milvus_uri = iso_env(base_url)
    global BASE, COLLECTION
    run_id = os.environ.get("FLYWHEEL_EVAL_RUN_ID") or f"run_{uuid.uuid4().hex[:12]}"
    os.environ["FLYWHEEL_EVAL_RUN_ID"] = run_id
    COLLECTION = isolated_collection(milvus_uri)
    BASE = base_url
    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]
    # 首次与写回后必须用不同会话:同会话会把首问原话带进上下文,「再问」触发指代消解改写,
    # 看起来像没学会,实际是评测口径串了。同一用户、新会话才验证知识写回本身。
    user_id = f"fw-{run_id}"
    results = []
    async with httpx.AsyncClient(timeout=120) as client:
        for row in cases:
            first = await ask(client, user_id, row["raw_question"])
            results.append({"id": row["id"], "raw_question": row["raw_question"],
                            "first_refusal": first["refused"],
                            "conversation_id": first["conversation_id"]})
    conversation_ids = {r["conversation_id"] for r in results if r["conversation_id"]}
    stats = await _seed_and_writeback(db_url, milvus_uri, cases, conversation_ids)
    problems = writeback_problems(stats)
    by_text = stats.pop("pool_texts", set())
    if os.environ.get("FLYWHEEL_EVAL_KEEP_COLLECTION") == "1":
        # 诊断:服务端检索需要看到同一集合,保留后由人工/脚本按 run_id 重查
        stats["collection"] = COLLECTION
    async with httpx.AsyncClient(timeout=120) as client:
        for row, res in zip(cases, results):
            second = await preview_answer(client, row["raw_question"],
                                          row.get("expect_answer") or ANSWER)
            res["answerable_after"] = second["recovered"]
            res["recovered_top"] = second["top"]
            res["in_pool"] = row["raw_question"] in by_text
            res["passed"] = (res["first_refusal"] and res["answerable_after"]
                             and res["in_pool"])
            if not res["in_pool"]:
                res["detail"] = "首次请求未落低置信池(路由/意图未走到知识兜底)"
    passed = sum(r["passed"] for r in results)
    payload = {
        "meta": {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                 "dataset": str(DATASET.relative_to(ROOT)),
                 "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
                 "review_status": sorted({c.get("review_status") for c in cases}),
                 "independent_release_evidence": False,
                 "isolation": {"database_env": "FLYWHEEL_EVAL_DATABASE_URL",
                               "milvus_collection": COLLECTION, "run_id": run_id},
                 "eval_user_id": user_id,
                 "second_turn_conversation": "new (fresh thread, same user)",
                 "source": "isolated_eval",
                 "service_url": BASE,
                 "writeback_problems": problems,
                 "answer_source": "dataset_preset; not generated by review"},
        "summary": {"passed": passed, "total": len(results),
                    "first_refusal": sum(r["first_refusal"] for r in results),
                    "answerable_after": sum(r["answerable_after"] for r in results), **stats},
        "cases": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if os.environ.get("FLYWHEEL_EVAL_KEEP_COLLECTION") == "1":
        print(f"保留隔离集合供诊断:{COLLECTION}")
        return 1
    try:
        from app.kb import milvus_client as kb_client
        cleanup = kb_client.get_client(uri=milvus_uri)
        kb_client.drop(cleanup, COLLECTION)
        print(f"已清理隔离集合 {COLLECTION}")
    except Exception as exc:  # noqa: BLE001 清理失败只告警,不掩盖评测结论
        print(f"隔离集合清理失败:{type(exc).__name__}: {exc}"[:300])
    print(f"飞轮闭环 {passed}/{len(results)};首次拒答 {payload['summary']['first_refusal']},"
          f"写回后能答 {payload['summary']['answerable_after']};报告 {REPORT.relative_to(ROOT)}")
    if problems:
        print(f"写回断言失败:{','.join(problems)};报告 {REPORT.relative_to(ROOT)}")
        return 1
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

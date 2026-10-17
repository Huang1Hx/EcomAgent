"""Task 3 数据合同:字段齐全、ID 唯一、split 不交叉、文本精确去重、支持数、manifest 哈希。"""
import hashlib
import json
from pathlib import Path

import pytest

EVAL = Path("data/eval")
MANIFESTS = EVAL / "manifests"
FILES = {
    "rag_dev.jsonl": ("rag", 300),
    "rag_blind_test.jsonl": ("rag", 300),
    "intent_test.jsonl": ("intent", 180),
    "agent_test.jsonl": ("agent", 120),
    "flywheel_closure_test.jsonl": ("flywheel", 50),
    "context_test.jsonl": ("context", 50),
    "confidence_calibration.jsonl": ("confidence", 120),
    "confidence_blind_test.jsonl": ("confidence", 120),
}
COMMON = {"id", "source", "created_at", "review_status", "split", "task"}
ALLOWED_REVIEW = {"ai_assisted_unreviewed", "human_reviewed", "legacy_unaudited"}


def load(name):
    return [json.loads(l) for l in (EVAL / name).read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.parametrize("name", FILES)
def test_file_contract(name):
    rows = load(name)
    task, minimum = FILES[name]
    assert len(rows) >= minimum, f"{name} 只有 {len(rows)} 条"
    assert len({r["id"] for r in rows}) == len(rows), "ID 必须唯一"
    for r in rows:
        assert COMMON <= set(r), f"{r.get('id')} 缺字段 {COMMON - set(r)}"
        assert r["task"] == task
        assert r["review_status"] in ALLOWED_REVIEW
        assert r["source"] and r["created_at"]
    text_key = {"flywheel": "raw_question", "context": None,
                "agent": "query"}.get(task, "query")
    if text_key:
        texts = [r.get(text_key) for r in rows if r.get(text_key)]
        assert len(set(texts)) == len(texts), "文本必须精确去重"


def _queries(name):
    return {r["query"] for r in load(name)}


def test_rag_dev_and_blind_do_not_cross():
    dev, blind = _queries("rag_dev.jsonl"), _queries("rag_blind_test.jsonl")
    assert not (dev & blind)
    assert all(r["split"] == "dev" for r in load("rag_dev.jsonl"))
    assert all(r["split"] == "blind" for r in load("rag_blind_test.jsonl"))


def test_confidence_calibration_and_blind_are_isolated():
    cal, blind = _queries("confidence_calibration.jsonl"), _queries("confidence_blind_test.jsonl")
    assert not (cal & blind)
    assert {r["split"] for r in load("confidence_calibration.jsonl")} == {"calibration"}
    assert {r["split"] for r in load("confidence_blind_test.jsonl")} == {"blind"}
    c = load("confidence_blind_test.jsonl")
    assert sum(not r["should_refuse"] for r in c) >= 120
    assert sum(r["should_refuse"] for r in c) >= 120


def test_rag_blind_bucket_support():
    rows = load("rag_blind_test.jsonl")
    counts = {}
    for r in rows:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
    for bucket in ("A_policy", "B_model", "C_colloquial", "E_multi", "D_absent"):
        assert counts[bucket] >= 60, (bucket, counts)


def test_intent_label_support():
    rows = load("intent_test.jsonl")
    counts = {}
    for r in rows:
        if r.get("expect_intent"):
            counts[r["expect_intent"]] = counts.get(r["expect_intent"], 0) + 1
    assert len(counts) == 9
    assert all(n >= 20 for n in counts.values()), counts
    assert sum(r["kind"] == "drift" for r in rows) >= 45 - 30  # 漂移+边界共45条
    assert sum(r["kind"] in ("drift", "boundary") for r in rows) >= 45


def test_agent_composition():
    rows = load("agent_test.jsonl")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    assert kinds["single"] >= 90 and kinds["toolchain"] >= 15 and kinds["multi"] >= 15


@pytest.mark.parametrize("name", list(FILES))
def test_manifest_matches_file(name):
    path = EVAL / name
    m = json.loads((MANIFESTS / f"{path.stem}.json").read_text(encoding="utf-8"))
    rows = load(name)
    assert m["dataset"] == str(path)
    assert m["samples"] == len(rows)
    assert m["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert m["reviewers"] == 0 and m["review_status"] == "ai_assisted_unreviewed"
    assert m["independent_release_evidence"] is False
    assert "support" in m and "dedup" in m and "generation_method" in m


def test_legacy_300_is_development_only():
    dev = json.loads((MANIFESTS / "rag_dev.json").read_text(encoding="utf-8"))
    assert dev["legacy_development_set"]["samples"] == 300
    assert dev["legacy_development_set"]["not_independent_test"] is True

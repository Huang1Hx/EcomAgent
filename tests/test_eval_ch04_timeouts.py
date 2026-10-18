"""RAG 评测的等待上限与可恢复小样:超时必须失败关闭,不能被当成题做对。"""
import asyncio
import json

import pytest


def test_default_call_timeout_is_60s_and_configurable(monkeypatch):
    from scripts import eval_ch04

    monkeypatch.delenv("RAG_CALL_TIMEOUT", raising=False)
    assert eval_ch04.eval_call_timeout() == 60.0
    monkeypatch.setenv("RAG_CALL_TIMEOUT", "45")
    assert eval_ch04.eval_call_timeout() == 45.0


def test_read_note_timeout_is_shorter_but_configurable(monkeypatch):
    from app.core import read_notes

    monkeypatch.delenv("RAG_NOTE_TIMEOUT", raising=False)
    assert read_notes.timeout_seconds() == 30.0
    assert read_notes.timeout_seconds() < 60.0
    monkeypatch.setenv("RAG_NOTE_TIMEOUT", "12")
    assert read_notes.timeout_seconds() == 12.0
    monkeypatch.setenv("RAG_NOTE_TIMEOUT", "0")
    with pytest.raises(ValueError):
        read_notes.timeout_seconds()


@pytest.mark.asyncio
async def test_retrieval_wait_is_bounded_and_reported_as_provider_error(monkeypatch):
    from scripts import eval_ch04

    monkeypatch.setenv("RAG_CALL_TIMEOUT", "0.02")

    async def hangs(*args, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(eval_ch04.retrieval, "search_knowledge", hangs)
    with pytest.raises(eval_ch04.ProviderCallError) as exc:
        await eval_ch04._retrieve("vector", {"id": "Q1", "query": "退款"})
    assert exc.value.error_type == "TimeoutError"


@pytest.mark.asyncio
async def test_read_note_timeout_returns_fallback_not_note(monkeypatch):
    from app.core import read_notes

    monkeypatch.setenv("RAG_NOTE_TIMEOUT", "0.01")

    class Slow:
        async def ainvoke(self, prompt):
            await asyncio.sleep(5)

    assert await read_notes.generate("rag_mrr", {"a": 1}, model=Slow()) is None


def _write(tmp_path, rows):
    path = tmp_path / "blind.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_eval_limit_takes_a_deterministic_small_sample(monkeypatch, tmp_path):
    from scripts import eval_ch04

    rows = [{"id": str(i), "bucket": "D_absent" if i == 1 else "A_policy",
             "query": f"q{i}"} for i in range(5)]
    path = _write(tmp_path, rows)
    monkeypatch.delenv("RAG_EVAL_LIMIT", raising=False)
    assert [s["id"] for s in eval_ch04.load_samples(str(path))] == list("01234")
    monkeypatch.setenv("RAG_EVAL_LIMIT", "3")
    assert [s["id"] for s in eval_ch04.load_samples(str(path))] == ["0", "1", "2"]
    monkeypatch.setenv("RAG_EVAL_LIMIT", "2")
    assert [s["id"] for s in eval_ch04.load_samples(str(path))] == ["0", "1"]
    for bad in ("0", "-1", "many"):
        monkeypatch.setenv("RAG_EVAL_LIMIT", bad)
        with pytest.raises(ValueError):
            eval_ch04.load_samples(str(path))

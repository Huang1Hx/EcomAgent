import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import agent_eval


@pytest.fixture
def report(tmp_path, monkeypatch):
    path = tmp_path / "agent_eval.json"
    monkeypatch.setattr(agent_eval, "REPORT", path)
    return path


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(agent_eval.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_missing_report_returns_rerun_hint(client, report):
    body = (await client.get("/api/agent-eval/overview")).json()
    assert body["present"] is False
    assert body["make"] == "make eval-agent"
    assert body["job"]["name"] == "eval-agent"


async def test_report_is_exposed_without_recomputing(client, report):
    payload = {
        "meta": {"chat_model": "test-model", "generated_at": "2026-09-08T12:00:00"},
        "summary": {"passed": 2, "total": 2, "rate": 1.0, "all_pass": True},
        "cases": [{"id": "A01", "passed": True}],
    }
    report.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    body = (await client.get("/api/agent-eval/overview")).json()
    assert body["present"] is True
    assert body["summary"] == payload["summary"]
    assert body["cases"] == payload["cases"]

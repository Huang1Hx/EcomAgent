import pytest

from app.core import query_understanding


@pytest.mark.asyncio
async def test_rewrite_provider_failure_keeps_safe_fallback_and_diagnostic(monkeypatch):
    class Upstream499(Exception):
        status_code = 499

    class Chain:
        async def ainvoke(self, _):
            raise Upstream499()

    class Prompt:
        def __or__(self, model):
            return Chain()

    monkeypatch.setattr(query_understanding.llm, "structured", lambda *args: object())
    monkeypatch.setattr(query_understanding, "QUERY_REWRITE_PROMPT", Prompt())
    out = await query_understanding.understand("邮费怎么算")
    assert out == {"standard": "邮费怎么算", "expanded": [],
                   "provider_error": "query_rewrite", "provider_status_code": 499}

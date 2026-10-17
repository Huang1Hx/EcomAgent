"""确定性答复在流式接口里只能发一次 delta:log_node 写回消息后不得再被当成模型输出。"""
import pytest

from app.graph import runtime


class _Graph:
    def __init__(self):
        self.calls = 0

    async def astream(self, _src, _cfg, stream_mode=None):
        self.calls += 1
        # 先发确定性节点的 answer 更新(前端靠这个出字),再发 log 的 messages 更新
        yield "updates", {"fallback_reply": {"answer": "兜底话术"}}
        yield "updates", {"log": {"messages": [object()]}}
        yield "updates", {"log": {}}


@pytest.mark.asyncio
async def test_deterministic_answer_emitted_once(monkeypatch):
    monkeypatch.setattr(runtime, "get_graph", lambda: _Graph())
    events = [ev async for ev in runtime._stream_events(1, {})]
    deltas = [ev for ev in events if ev["type"] == "delta"]
    assert [d["text"] for d in deltas] == ["兜底话术"]
    assert events[-1]["type"] == "done"

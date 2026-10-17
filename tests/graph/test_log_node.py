import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


def test_resolve_answer_prefers_explicit():
    assert nodes.resolve_answer({"answer": "固定话术", "messages": []}) == "固定话术"


def test_resolve_answer_falls_back_to_last_ai():
    state = {"messages": [HumanMessage("hi"), AIMessage("模型答复")]}
    assert nodes.resolve_answer(state) == "模型答复"


@pytest.mark.asyncio
async def test_log_node_persists_assistant(monkeypatch):
    saved = {}

    async def fake_append(cid, role, content=None, **k):
        saved.update(cid=cid, role=role, content=content)
        return 1

    monkeypatch.setattr(nodes.repository, "append_message", fake_append)
    await nodes.log_node({"conversation_id": 8, "intent": "订单", "route": "business",
                          "messages": [AIMessage("订单已发货")], "trace": {"forced_rag": False}})
    assert saved["cid"] == 8
    assert saved["role"] == "assistant"
    assert saved["content"] == "订单已发货"


@pytest.mark.asyncio
async def test_log_node_adds_deterministic_reply_to_graph_history(monkeypatch):
    """确定性答复落库后必须回到 messages:否则下一轮窗口里是两条连续 HumanMessage。"""
    async def fake_append(cid, role, content=None, **k):
        return 77

    monkeypatch.setattr(nodes.repository, "append_message", fake_append)
    out = await nodes.log_node({"conversation_id": 8, "answer": "兜底话术",
                                "messages": [HumanMessage("今天天气怎么样")]})
    assert len(out["messages"]) == 1
    msg = out["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == "兜底话术"
    assert msg.id == "db-77"


@pytest.mark.asyncio
async def test_log_node_does_not_duplicate_agent_reply(monkeypatch):
    """Agent 路径的 AIMessage 已在 messages 里且 answer 为空:不得再返回一条。"""
    async def fake_append(cid, role, content=None, **k):
        return 78

    monkeypatch.setattr(nodes.repository, "append_message", fake_append)
    out = await nodes.log_node({"conversation_id": 8,
                                "messages": [HumanMessage("hi"), AIMessage("模型答复")]})
    assert not out.get("messages")

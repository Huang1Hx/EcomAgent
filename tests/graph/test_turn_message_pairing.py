"""跨轮消息配对回归(隔离图 + 假模型,不碰真实上游/数据库)。

复现:确定性答复只写 state.answer,没有写回 messages,导致下一轮模型窗口里出现
连续两条 HumanMessage,模型把上一轮问题和本轮问题一起回答。
"""
from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build as build_mod
from app.graph import routing
from app.graph import nodes
from app.graph import nodes as real_nodes


class ScriptedModel(BaseChatModel):
    """回放固定 AIMessage,并记录每次收到的完整消息窗口。"""
    replies: list[str]
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self.replies.pop(0)))])


INTENTS = {"今天天气怎么样": "其他", "我的订单到哪了": "物流"}


async def _classify_intent(state):
    """天气→其他(知识路弱证据),订单→物流(business)。"""
    return {"intent": INTENTS[state["messages"][-1].content],
            "resolved_query": state["messages"][-1].content}


async def _resolve_reference(state):
    return {}


async def _retrieve_knowledge(state):
    return {"evidence": "", "evidence_strong": False, "citations": [],
            "trace": {"route": "knowledge"}}


async def _fallback_reply(state):
    return {"answer": "这个问题我暂时没有查到确切信息。",
            "trace": {"route": "fallback_script"}}


@pytest.fixture()
def fake_nodes(monkeypatch):
    model = ScriptedModel(replies=["已为您查询订单物流。"])

    async def main_agent(state):
        window = real_nodes._agent_messages(state)
        return {"messages": [await model.ainvoke(window)]}

    saved = []

    async def fake_append(cid, role, content=None, **k):
        saved.append({"role": role, "content": content})
        return len(saved)

    monkeypatch.setattr(real_nodes.repository, "append_message", fake_append)
    nodes = SimpleNamespace(**{
        **vars(real_nodes),
        "classify_intent": _classify_intent,
        "resolve_reference": _resolve_reference,
        "retrieve_knowledge": _retrieve_knowledge,
        "fallback_reply": _fallback_reply,
        "main_agent": main_agent,
    })

    def router(state):
        return routing.INTENT_TO_ROUTE.get(state.get("intent", ""), "business")

    def builder():
        return build_mod._builder(nodes, router).compile(checkpointer=InMemorySaver())

    return SimpleNamespace(builder=builder, model=model, saved=saved)


async def _turn(graph, text, cfg):
    await graph.ainvoke({"messages": [HumanMessage(text)], "conversation_id": 1,
                         "user_id": "u", "answer": "", "trace": None,
                         "summary_upto_msg_id": 0, "layer1_from_msg_id": 0,
                         # 与 runtime._graph_input 一致:本轮输出通道显式清零
                         "intent": "", "resolved_query": ""}, cfg)


@pytest.mark.asyncio
async def test_fallback_then_business_turn_keeps_question_answer_pairs(fake_nodes):
    graph = fake_nodes.builder()
    cfg = {"configurable": {"thread_id": "pairing-1"}}

    await _turn(graph, "今天天气怎么样", cfg)
    state = await graph.aget_state(cfg)
    roles = [m.type for m in state.values["messages"]]
    assert roles == ["human", "ai"], f"拒答轮后应是一问一答,实际 {roles}"

    calls_before = len(fake_nodes.model.seen)
    await _turn(graph, "我的订单到哪了", cfg)
    # 取本轮第一次 Agent 调用的窗口:它在看到本轮新问题时,历史必须已是一问一答
    window = fake_nodes.model.seen[calls_before]
    window_roles = [m.type for m in window]
    assert window_roles == ["system", "human", "ai", "human"], window_roles
    # 第一轮是"其他"意图的脚本答复(确定性答复的一种),必须作为 AI 消息出现在历史里
    assert window[-2].content == nodes.SCRIPT_REPLY_OTHER
    assert window[-1].content == "我的订单到哪了"

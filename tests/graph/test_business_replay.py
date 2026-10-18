"""Task 2 业务回放:真实前置节点 + 脚本化 Agent,验证只读商品/本人订单与外单不可见。"""
from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build as build_mod
from app.graph import nodes as real_nodes
from app.graph import routing


class ScriptedModel(BaseChatModel):
    seen: list = []

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage("已根据本轮事实作答。"))])


@pytest.fixture()
def replay(monkeypatch):
    model = ScriptedModel()
    intents = {"你们卖什么": "商品咨询", "查看我的订单": "订单",
               "订单 2001 的物流到哪了": "物流"}

    async def classify(state):
        text = state["messages"][-1].content
        return {"intent": intents[text], "resolved_query": text}

    async def resolve(state):
        return {}

    async def agent(state):
        return {"messages": [await model.ainvoke(real_nodes._agent_messages(state))]}

    async def append(cid, role, content=None, **k):
        return 1

    monkeypatch.setattr(real_nodes.repository, "append_message", append)
    async def knowledge(state):
        return {"evidence": "", "evidence_strong": False, "citations": [],
                "trace": {"route": "knowledge"}}

    async def fallback(state):
        return {"answer": "这个问题我暂时没有查到确切信息。", "trace": {"route": "fallback_script"}}

    fake = SimpleNamespace(**{**vars(real_nodes), "classify_intent": classify,
                              "resolve_reference": resolve, "main_agent": agent,
                              "retrieve_knowledge": knowledge, "fallback_reply": fallback})

    def router(state):
        return routing.route_by_intent(state)

    graph = build_mod._builder(fake, router).compile(checkpointer=InMemorySaver())
    return SimpleNamespace(graph=graph, model=model)


async def _turn(replay, text, user_id, tid):
    cfg = {"configurable": {"thread_id": tid}}
    await replay.graph.ainvoke({"messages": [HumanMessage(text)], "conversation_id": 1,
                                "user_id": user_id, "answer": "", "trace": None,
                                "summary_upto_msg_id": 0, "layer1_from_msg_id": 0,
                                "intent": "", "resolved_query": ""}, cfg)
    return await replay.graph.aget_state(cfg)


@pytest.mark.asyncio
async def test_catalog_question_uses_local_catalog(replay):
    state = await _turn(replay, "你们卖什么", "u-alice", "biz-catalog")
    ctx = state.values["business_context"]
    assert ctx["kind"] == "products" and ctx["products"]


@pytest.mark.asyncio
async def test_my_orders_are_limited_to_the_current_user(replay):
    state = await _turn(replay, "查看我的订单", "u-alice", "biz-orders")
    ctx = state.values["business_context"]
    ids = {o["order_id"] for o in ctx["orders"]}
    assert {"1001", "2002"} <= ids


@pytest.mark.asyncio
async def test_foreign_order_logistics_is_not_exposed(replay):
    state = await _turn(replay, "订单 2001 的物流到哪了", "u-alice", "biz-foreign")
    ctx = state.values["business_context"]
    assert ctx["kind"] == "order_not_owned"
    assert "tracking_no" not in state.values.get("order_data", {})

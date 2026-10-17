"""business 出口在进 Agent 前确定性准备事实;Agent 不得绕过前置节点。"""
import pytest
from langchain_core.messages import HumanMessage

from app.graph import build as build_mod
from app.graph import nodes
from app.graph.build import _builder


def test_business_routes_through_prepare_business_context():
    g = _builder().compile()
    edges = {(e.source, e.target) for e in g.get_graph().edges}
    assert ("prepare_business_context", "main_agent") in edges
    assert ("classify_intent", "main_agent") not in {
        (e.source, e.target) for e in g.get_graph().edges if e.source == "classify_intent"
    }


@pytest.mark.asyncio
async def test_prepare_products_for_catalog_question(monkeypatch):
    out = await nodes.prepare_business_context(
        {"messages": [HumanMessage("你们卖什么")], "resolved_query": "你们卖什么",
         "user_id": "u-alice", "intent": "商品咨询", "trace": {}})
    assert out["business_context"]["kind"] == "products"
    assert out["business_context"]["products"]
    assert out["trace"]["business_context"] == "products"


@pytest.mark.asyncio
async def test_prepare_orders_lists_current_user(monkeypatch):
    out = await nodes.prepare_business_context(
        {"messages": [HumanMessage("查看我的订单")], "resolved_query": "查看我的订单",
         "user_id": "u-alice", "intent": "订单", "trace": {}})
    assert out["business_context"]["kind"] == "orders"
    ids = {o["order_id"] for o in out["business_context"]["orders"]}
    assert {"1001", "2002"} <= ids
    assert out["order_data"] == {}
    assert out["suggested_actions"][0]["type"] == "select_order"


@pytest.mark.asyncio
async def test_prepare_logistics_blocks_foreign_order(monkeypatch):
    """用户报了别人的单号:前置节点不能把他人订单/物流单号交给 Agent。"""
    from app.tools import business
    mine = {o["order_id"] for o in business.list_user_orders("u-alice")}
    other = next(o["order_id"] for o in business.list_user_orders("u-bob")
                 if o["order_id"] not in mine)
    out = await nodes.prepare_business_context(
        {"messages": [HumanMessage(f"订单{other}的物流到哪了")],
         "resolved_query": f"订单{other}的物流到哪了", "user_id": "u-alice",
         "intent": "物流", "trace": {}})
    ctx = out["business_context"]
    assert ctx["kind"] == "order_not_owned"
    assert ctx["order_id"] == other
    assert "tracking_no" not in out.get("order_data", {})
    assert "没有找到" in ctx["message"]


@pytest.mark.asyncio
async def test_prepare_own_order_logistics_has_tracking(monkeypatch):
    out = await nodes.prepare_business_context(
        {"messages": [HumanMessage("查订单1001物流")], "resolved_query": "查订单1001物流",
         "user_id": "u-alice", "intent": "物流", "trace": {}})
    assert out["business_context"]["kind"] == "logistics"
    assert out["order_data"]["tracking_no"].startswith("SF")


@pytest.mark.asyncio
async def test_agent_window_includes_business_facts():
    ctx = await nodes.prepare_business_context(
        {"messages": [HumanMessage("你们卖什么")], "resolved_query": "你们卖什么",
         "user_id": "u-alice", "intent": "商品咨询", "trace": {}})
    ctx.update({"messages": [HumanMessage("你们卖什么")], "route": "business"})
    msgs = nodes._agent_messages(ctx)
    turn_ctx = next(m for m in msgs if getattr(m, "id", None) == nodes.TURN_CTX_ID)
    assert "本轮确定性业务事实" in turn_ctx.content
    assert "TY-WAND" in turn_ctx.content
    assert "不能代" in turn_ctx.content or "不能代下单" in turn_ctx.content

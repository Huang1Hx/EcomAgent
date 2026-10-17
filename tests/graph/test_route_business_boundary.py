"""Task 2 路由边界:浏览/购买诉求走 business,选品知识仍走 RAG。"""
import pytest

from app.graph import routing
from app.graph.routing import route_by_intent


@pytest.mark.parametrize("query", ["你们卖什么", "有什么玩具", "猫粮有哪些", "我要买幼猫粮",
                                   "帮我下单", "查看我的订单", "我的订单"])
def test_catalog_and_order_requests_go_business(query):
    assert route_by_intent({"intent": "商品咨询", "resolved_query": query}) == "business"
    assert route_by_intent({"intent": "订单", "resolved_query": query}) == "business"


def test_order_listing_helper():
    assert routing.is_order_listing("查看我的订单")
    assert routing.is_order_listing("我有哪些订单")
    assert not routing.is_order_listing("订单 1001 到哪了")


@pytest.mark.parametrize("query", ["猫粮怎么选", "猫粮怎么保存"])
def test_knowledge_questions_stay_rag(query):
    assert route_by_intent({"intent": "商品咨询", "resolved_query": query}) == "knowledge"


def test_price_question_goes_catalog_business():
    assert route_by_intent({"intent": "商品咨询", "resolved_query": "汽水多少钱"}) == "business"


@pytest.mark.asyncio
async def test_prepared_facts_drive_reply_not_knowledge_refusal():
    """“你们卖什么”经 business 前置节点拿到真实目录,答案不再走知识库拒答。"""
    from langchain_core.messages import HumanMessage
    from app.graph import nodes
    out = await nodes.prepare_business_context(
        {"messages": [HumanMessage("你们卖什么")], "resolved_query": "你们卖什么",
         "user_id": "u-alice", "intent": "商品咨询", "trace": {}})
    products = out["business_context"]["products"]
    assert len(products) == 25
    assert out["business_context"]["kind"] == "products"

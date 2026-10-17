"""本人订单列表:只返回注入身份名下的订单,不做跨用户枚举。"""
import pytest

from app.tools import business
from app.tools.builtin.orders import list_orders


async def test_list_orders_only_own_orders():
    out = await list_orders.ainvoke({"user_id": "u-alice"})
    ids = {o["order_id"] for o in out["orders"]}
    assert ids == {o["order_id"] for o in business.list_user_orders("u-alice")}
    assert {"1001", "2002"} <= ids
    bob = {o["order_id"] for o in business.list_user_orders("u-bob")}
    alice_private = ids - set(bob)
    assert not (alice_private & {o["order_id"] for o in (await list_orders.ainvoke({"user_id": "u-bob"}))["orders"]})


async def test_list_orders_empty_identity_returns_empty():
    out = await list_orders.ainvoke({"user_id": ""})
    assert out == {"orders": [], "count": 0, "message": "缺少身份信息,无法列出订单"}


async def test_list_orders_stable_and_shaped():
    a = await list_orders.ainvoke({"user_id": "u-alice"})
    b = await list_orders.ainvoke({"user_id": "u-alice"})
    assert a == b
    assert all({"order_id", "product", "status", "amount"} <= set(o) for o in a["orders"])

from app.tools import business
from app.tools.builtin.orders import query_order, query_product

USER = "u-mock"
MINE = business.list_user_orders(USER)[0]["order_id"]   # 这个用户名下真实存在的一单

# ch08:query_logistics 内置版下线,物流查询由物流 MCP Server 接管(见 tests/tools/test_mcp_servers.py)


async def test_query_order_deterministic_and_shaped():
    # ch08:身份由执行引擎注入,查自己的单才有数据(归属校验见 tests/tools/test_order_ownership.py)
    r1 = await query_order.ainvoke({"order_id": MINE, "user_id": USER})
    r2 = await query_order.ainvoke({"order_id": MINE, "user_id": USER})
    assert r1 == r2                                  # 同种子可复现
    assert r1["order_id"] == MINE
    assert r1["status"] in {"待付款", "已付款", "已发货", "已签收"}
    assert r1["tracking_no"].startswith("SF")        # 物流单号,查物流需先拿它(ch05 造真实依赖)


async def test_query_product_names_and_determinism():
    assert query_order.name == "query_order"
    assert query_product.name == "query_product"
    p1 = await query_product.ainvoke({"product_name": "猫粮"})
    p2 = await query_product.ainvoke({"product_name": "猫粮"})
    assert p1 == p2                                   # 同种子可复现
    assert p1["found"] is True
    assert p1["match_count"] >= 2
    assert all("price" in product and "stock" in product for product in p1["products"])


async def test_query_product_目录外商品明确返回未找到():
    out = await query_product.ainvoke({"product_name": "汽水"})
    assert out == {
        "found": False,
        "query": "汽水",
        "code": "product_not_found",
        "message": "商品目录中未找到“汽水”，当前不销售该商品。",
    }


async def test_query_product_具体品类不混入更短的相关词商品():
    out = await query_product.ainvoke({"product_name": "智能猫砂盆"})
    assert {product["sku"] for product in out["products"]} == {
        "MH-LP100", "MH-LP50", "MH-LP200",
    }


async def test_query_product_缺货商品与不存在商品语义不同():
    out = await query_product.ainvoke({"product_name": "宠物定位器"})
    assert out["found"] is True
    assert out["products"][0]["stock"] == 0
    assert out["products"][0]["in_stock"] is False


async def test_list_products_and_list_orders_registered():
    from app.tools import registry
    by = {s.name: s for s in registry.builtin_specs()}
    assert {"list_products", "list_orders"} <= set(by)
    assert by["list_products"].inject_user_id is False
    assert by["list_orders"].inject_user_id is True
    assert all(s.permission == "read" for s in (by["list_products"], by["list_orders"]))
    props = by["list_orders"].json_schema.get("properties", {})
    assert "user_id" not in props   # 身份由引擎注入,模型不可见


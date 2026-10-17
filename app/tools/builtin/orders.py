from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from app.tools import registry
from app.tools.business import list_user_orders, order_snapshot, owns_order
from app.tools.catalog import browse_products, categories, public_product, search_products

# 查不到和不是本人的单回同一句话。分开回就成了枚举 oracle:攻击者靠回答差异
# 就能挨个试出哪些订单号真实存在。
NOT_OWNED = {"error": "没有找到您的这笔订单", "code": "order_not_owned"}


@tool
async def query_order(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
    user_id: Annotated[str, InjectedToolArg],
) -> dict:
    """查询订单的状态、金额、下单时间、商品名和物流单号(tracking_no)。用于用户询问某个订单情况时。
    要查物流轨迹,需先用本工具拿到订单的 tracking_no,再把它传给 query_logistics。
    发起人身份由系统注入,你不要传 user_id。"""
    if not owns_order(user_id, order_id):
        return dict(NOT_OWNED)
    return order_snapshot(order_id)


@tool
async def query_product(
    product_name: Annotated[str, Field(description="商品名称或关键词,例如 猫粮")],
) -> dict:
    """查询目录中真实存在商品的价格、库存和规格。

    用户问是否销售、是否有货、多少钱时必须调用。found=false 表示目录中没有该商品，
    应明确告知当前不销售，严禁根据商品名自行生成价格或库存。
    """
    products = [public_product(p) for p in search_products(product_name)]
    if not products:
        return {
            "found": False,
            "query": product_name,
            "code": "product_not_found",
            "message": f"商品目录中未找到“{product_name}”，当前不销售该商品。",
        }
    return {"found": True, "query": product_name,
            "match_count": len(products), "products": products}


@tool
async def list_products(
    category: Annotated[str | None, Field(description="品类,例如 玩具、宠物食品;不传则列出全部品类")] = None,
) -> dict:
    """浏览店铺实际在售的商品目录。用户问“你们卖什么”“有什么玩具”“猫粮有哪些”时调用。
    只返回目录里真实存在的商品;目录里没有的品类返回空列表并明确说当前不销售,
    严禁自己编造商品、价格或库存。"""
    products = [public_product(p) for p in browse_products(category)]
    all_categories = categories()
    if not products:
        what = category or "该品类"
        return {"categories": all_categories, "products": [], "total": 0,
                "code": "category_empty",
                "message": f"商品目录中没有“{what}”,当前不销售该类商品。"}
    return {"categories": all_categories, "products": products, "total": len(products)}


@tool
async def list_orders(
    user_id: Annotated[str, InjectedToolArg],
) -> dict:
    """列出当前用户自己名下的订单(订单号、商品、状态、金额)。用户问“我的订单”时调用。
    身份由系统注入,你不要传 user_id;只能看到本人订单,看不到别人的。"""
    if not user_id:
        return {"orders": [], "count": 0, "message": "缺少身份信息,无法列出订单"}
    rows = list_user_orders(user_id)
    return {"orders": rows, "count": len(rows)}


registry.register(registry.spec_from_langchain_tool(
    query_order, source="builtin", inject_user_id=True))
registry.register(registry.spec_from_langchain_tool(query_product, source="builtin"))
registry.register(registry.spec_from_langchain_tool(list_products, source="builtin"))
registry.register(registry.spec_from_langchain_tool(
    list_orders, source="builtin", inject_user_id=True))

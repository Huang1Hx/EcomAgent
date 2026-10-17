"""商品目录浏览:真实目录数据,不按输入编造;目录外明确不销售。"""
import pytest

from app.tools import catalog
from app.tools.builtin.orders import list_products, query_product


def test_browse_products_all_returns_categories_and_items():
    out = catalog.browse_products()
    assert len(out) == len(catalog.load_products())
    assert "玩具" in {p["category"] for p in out}


@pytest.mark.parametrize("category,keyword", [("玩具", "玩具"), ("宠物食品", "猫粮")])
def test_browse_products_by_category(category, keyword):
    out = catalog.browse_products(category)
    assert out, category
    assert all(p["category"] == category for p in out)
    # 宠物食品的“猫粮有哪些”必须真在目录里;玩具按品类本身即可
    if keyword == "猫粮":
        assert any("猫粮" in p["name"] for p in out)


def test_browse_products_unknown_category_is_empty():
    assert catalog.browse_products("汽水") == []


def test_browse_products_limit():
    assert len(catalog.browse_products(limit=3)) == 3


async def test_list_products_tool_shape():
    out = await list_products.ainvoke({"category": None})
    assert out["categories"] and out["products"] and out["total"] > 0
    assert set(out["products"][0]) >= {"sku", "product_name", "price", "stock", "in_stock"}


async def test_list_products_category_and_empty():
    out = await list_products.ainvoke({"category": "玩具"})
    assert out["products"] and all(p["category"] == "玩具" for p in out["products"])
    miss = await list_products.ainvoke({"category": "汽水"})
    assert miss["products"] == [] and miss["code"] == "category_empty"
    assert "汽水" in miss["message"]


async def test_soda_still_reports_not_sold():
    out = await query_product.ainvoke({"product_name": "汽水"})
    assert out["found"] is False and out["code"] == "product_not_found"

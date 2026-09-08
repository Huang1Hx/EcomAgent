"""受版本控制的演示商品目录。

商品查询必须有明确边界：目录里有才返回价格和库存，目录外统一返回 found=false。
这比按用户输入随机生成一件“商品”更接近真实电商系统，也能阻止不存在商品的幻觉。
"""
import json
import re
from functools import lru_cache
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "catalog" / "products.json"


def _norm(text: str) -> str:
    return re.sub(r"[\s\-_]+", "", (text or "").strip().casefold())


@lru_cache(maxsize=1)
def load_products() -> tuple[dict, ...]:
    rows = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("商品目录顶层必须是数组")
    required = {"sku", "name", "aliases", "category", "price", "stock", "spec", "description"}
    products: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError("商品目录存在缺字段记录")
        if row["sku"] in seen:
            raise ValueError(f"商品 SKU 重复:{row['sku']}")
        seen.add(row["sku"])
        products.append(row)
    return tuple(products)


def search_products(query: str, limit: int = 5) -> list[dict]:
    """精确名称/SKU/别名优先，其次做包含匹配；结果顺序与目录一致。"""
    needle = _norm(query)
    if not needle:
        return []

    primary_exact: list[dict] = []
    matches: list[dict] = []
    for product in load_products():
        primary = [_norm(product["sku"]), _norm(product["name"])]
        aliases = [_norm(name) for name in product.get("aliases", [])]
        if needle in primary:
            primary_exact.append(product)
        elif needle in aliases or any(needle in name for name in [*primary, *aliases]):
            matches.append(product)
    return (primary_exact or matches)[:limit]


def public_product(product: dict) -> dict:
    """只返回客服回答所需字段，不把内部别名配置暴露给模型。"""
    return {
        "sku": product["sku"],
        "product_name": product["name"],
        "category": product["category"],
        "price": product["price"],
        "stock": product["stock"],
        "in_stock": product["stock"] > 0,
        "spec": product["spec"],
        "description": product["description"],
    }

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from app.config import settings

# 九类意图 → 五出口(写死的分流规则,spec §3.1 / README route_by_intent;
# 返回值 = build.py 条件边映射键,单一来源不漂移)
INTENT_TO_ROUTE: dict[str, str] = {
    "投诉": "escalate",
    "闲聊": "fallback_script",
    "其他": "fallback_script",
    "商品咨询": "knowledge",
    "退款退货": "refund_flow",
    "售后": "refund_flow",
    "人工": "business",   # ch08:明确要求建工单/转人工 → 主力 Agent 走 create_ticket 确认流
    "物流": "business",
    "订单": "business",
}

# 浏览店铺在售商品:目录类问题不能走知识 RAG(知识库里没有商品清单),也不该被当成
# 「商品咨询」去查规格。命中即进 business,由 list_products 读版本控制目录回答。
_CATALOG_BROWSE_WORDS = (
    "卖什么", "卖哪些", "卖的有哪些", "有什么卖", "在卖什么", "都卖什么", "都卖哪些",
    "有哪些商品", "有什么商品", "有哪些产品", "有什么产品",
)
_CATALOG_CATEGORY_WORDS = ("玩具", "猫粮", "狗粮", "猫砂", "用品", "商品", "产品")
_ORDER_LIST_WORDS = ("我的订单", "查看订单", "订单列表", "查我的订单", "我有哪些订单",
                     "我的全部订单", "我的单子")
_BUY_WORDS = ("我要买", "我想买", "帮我下单", "直接下单", "给我下单", "买一个", "买一袋")


def route_by_intent(state) -> str:
    """按意图分流;未知意图保守归 business(让 Agent 自己应对)。"""
    intent = state.get("intent", "")
    query = _query_text(state)
    if intent == "商品咨询":
        if is_catalog_browse(query) or is_product_lookup(query):
            return "business"
    if is_order_listing(query):
        return "business"
    if is_buy_request(query):
        return "business"
    return INTENT_TO_ROUTE.get(intent, "business")


_PRODUCT_LOOKUP_WORDS = (
    "有货", "有没有", "还有货", "库存", "现货", "缺货", "补货",
    "卖不卖", "是否售卖", "在售", "多少钱", "价格", "售价", "能买吗", "买得到",
)


def _query_text(state) -> str:
    resolved = state.get("resolved_query")
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content.strip()
    return ""


def is_product_lookup(query: str) -> bool:
    """库存/价格/是否销售需要查商品目录；说明、规格与政策继续走知识库。"""
    compact = "".join((query or "").split())
    if any(word in compact for word in _PRODUCT_LOOKUP_WORDS):
        return True
    return compact.startswith("有") and compact.rstrip("?？").endswith("吗") and "什么" not in compact


def is_catalog_browse(query: str) -> bool:
    """「你们卖什么/有什么玩具/猫粮有哪些」= 浏览目录,不是查某件商品规格。"""
    compact = "".join((query or "").split()).rstrip("?？。!")
    if any(word in compact for word in _CATALOG_BROWSE_WORDS):
        return True
    # 「有什么X」:X 是品类词(玩具/猫粮…),问的是目录里有没有这一类
    if compact.startswith("有什么") and any(w in compact for w in _CATALOG_CATEGORY_WORDS):
        return True
    # 「X有哪些」:有品类词 + 有哪些
    if compact.endswith("有哪些") and any(w in compact for w in _CATALOG_CATEGORY_WORDS):
        return True
    return False


def is_order_listing(query: str) -> bool:
    compact = "".join((query or "").split())
    return any(word in compact for word in _ORDER_LIST_WORDS)


def is_buy_request(query: str) -> bool:
    compact = "".join((query or "").split())
    return any(word in compact for word in _BUY_WORDS)


def confidence_gate(state) -> str:
    """知识路生成前证据闸:强放行、弱兜底。"""
    return "strong" if state.get("evidence_strong") else "weak"


def should_continue(state) -> str:
    """ReAct 停止条件:无 tool_calls 就收敛,步数封顶则强停,否则继续。

    只用步数封顶,不在这里卡 token。防打转靠的是步数:一步一次模型调用,数得清、好解释,
    各家 Agent SDK 给的也是 max_turns / max_iterations 这类步数上限。
    token 花销是另一件事——它管的是烧钱和撑爆上下文,不是死循环,阈值要按真实用量分布标定,
    归第 9 章的成本控制一起讲。混在这里卡,一次估偏就把工具链整条掐断,而且掐在模型
    已经决定调工具之后,用户只看到半句「我这就去查」。"""
    if state.get("answer"):
        return "stop"   # 确定性出口(如 agent_tools 证据不足转兜底)已给答复,不再回模型
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if not has_tool_calls:
        return "stop"
    if state.get("steps", 0) >= settings.max_agent_steps:
        return "stop"
    return "continue"

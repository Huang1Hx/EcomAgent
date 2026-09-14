import re

from app.core.llm import get_chat_model
from app.core.prompts import COREF_REWRITE_PROMPT

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def clean_rewrite_output(text: str, fallback: str) -> str:
    """清掉兼容上游偶尔混进 content 的思考标签，只保留最终改写。"""
    raw = (text or "").strip()
    if not raw:
        return fallback
    # 观察到过只有闭标签的形态：思考正文</think>最终答案。此时末段才是有效输出。
    parts = re.split(r"</think>", raw, flags=re.IGNORECASE)
    if len(parts) > 1:
        raw = parts[-1]
    raw = _THINK_BLOCK.sub("", raw)
    raw = _THINK_TAG.sub("", raw).strip()
    return raw or fallback


async def resolve(query: str, history: str = "") -> str:
    """指代消解 + 口语归一。结合历史把半截话补成完整问句;已完整则原样。
    出错兜底:返回原句(不阻断下游)。"""
    try:
        model = get_chat_model()
        r = await (COREF_REWRITE_PROMPT | model).ainvoke(
            {"query": query, "history": history or "(无)"})
        text = clean_rewrite_output(
            r.content if isinstance(r.content, str) else "", query)
    except Exception:
        return query
    return text

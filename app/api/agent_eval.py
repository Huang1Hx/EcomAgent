"""主 Agent 端到端评测报告 API。页面只读 ``make eval-agent`` 落下的 JSON。"""
import json
from pathlib import Path

from fastapi import APIRouter

from app.core import jobs

router = APIRouter(prefix="/api/agent-eval")
REPORT = Path(__file__).resolve().parents[2] / "data" / "ch02" / "reports" / "agent_eval.json"


def _read() -> dict | None:
    try:
        value = json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


@router.get("/overview")
async def overview() -> dict:
    report = _read()
    job = jobs.status("eval-agent", with_log=True)
    if report is None:
        return {"present": False, "make": "make eval-agent", "job": job,
                "hint": "还没有主 Agent 评测报告，点击重跑后会检查工具选择和关键业务行为。"}
    return {**report, "present": True, "make": "make eval-agent", "job": job}

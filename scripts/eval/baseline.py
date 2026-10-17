"""Task 0/6 基线采集:只读记录 Git、模型配置、知识库块数、评测集哈希与服务状态。
产物 data/eval/manifests/baseline.json 供后续报告比较;不输出任何密钥内容,只记是否存在。

用法:
  PYTHONPATH=. uv run python -m scripts.eval.baseline            # 静态信号(不调上游)
  PYTHONPATH=. uv run python -m scripts.eval.baseline --replay   # 追加在线流程回放(需 :8000)
  PYTHONPATH=. uv run python -m scripts.eval.baseline --tests    # 追加全量单测计数
"""
import argparse
import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys

import httpx

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/eval/manifests/baseline.json"
API = "http://127.0.0.1:8000"


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def models() -> dict:
    """只记录模型名/地址,密钥只留布尔。"""
    env = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
            if m:
                env[m.group(1)] = m.group(2)
    keys = {}
    for name, val in env.items():
        if name.endswith("_KEY"):
            keys[name] = bool(val.strip())
    return {
        "chat_model": env.get("CHAT_MODEL", ""),
        "chat_base_url": env.get("CHAT_BASE_URL", ""),
        "intent_model": env.get("INTENT_MODEL", "") or env.get("CHAT_MODEL", ""),
        "embed_model": "BAAI/bge-m3",
        "rerank_model": "BAAI/bge-reranker-v2-m3",
        "keys_present": dict(sorted(keys.items())),
        "thresholds": {
            "evidence_confidence_threshold": _settings_value("evidence_confidence_threshold"),
            "rerank_min_score": _settings_value("rerank_min_score"),
            "intent_conf_threshold": _settings_value("intent_conf_threshold"),
        },
    }


def _settings_value(name: str):
    sys.path.insert(0, str(ROOT))
    from app.config import settings
    return getattr(settings, name, None)


def kb_chunks() -> dict:
    sys.path.insert(0, str(ROOT))
    from app.kb import milvus_client
    try:
        client = milvus_client.get_client()
        return {"collection": milvus_client.COLLECTION, "vectors": milvus_client.count(client)}
    except Exception as exc:  # 服务不可用如实记录,不写成 0
        return {"error": f"{type(exc).__name__}: {exc}"}


def datasets() -> dict:
    out = {}
    paths = sorted((ROOT / "tests/data").glob("*.jsonl")) + sorted((ROOT / "data/eval").glob("*.jsonl"))
    for path in paths:
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[str(path.relative_to(ROOT))] = {"samples": len(rows), "sha256": sha256(path)}
    return out


def services() -> dict:
    out = {}
    for name, url in (("app", f"{API}/docs"), ("classifier", "http://127.0.0.1:8110/healthz"),
                      ("logistics_mcp", "http://127.0.0.1:8101/mcp"),
                      ("aftersales_mcp", "http://127.0.0.1:8102/mcp")):
        try:
            with httpx.Client(timeout=3) as c:
                r = c.get(url)
            out[name] = {"up": True, "status": r.status_code}
            if name == "classifier":
                body = r.json()
                out[name]["bundle_sha256"] = body.get("bundle_sha256")
        except Exception as exc:
            out[name] = {"up": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


REPLAY = [
    ("商品浏览", "你们卖什么"),
    ("我的订单", "查看我的订单"),
    ("知识拒答", "今天天气怎么样"),
    ("拒答后订单查询", "我的订单到哪了"),
]


async def replay(user_id: str = "baseline-probe") -> list[dict]:
    """固定 user_id + 新会话,逐轮回放;保存请求与结构化响应(不含密钥)。"""
    turns, cid = [], None
    async with httpx.AsyncClient(timeout=120) as client:
        for label, message in REPLAY:
            req = {"user_id": user_id, "message": message, "conversation_id": cid}
            r = await client.post(f"{API}/api/agent", json=req)
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            cid = body.get("conversation_id", cid)
            turns.append({"label": label, "request": req, "status": r.status_code,
                          "response": body})
    return turns


def tests() -> dict:
    proc = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "--tb=no"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-3:]
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", proc.stdout)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", proc.stdout)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", proc.stdout)) else 0
    skipped = int(m.group(1)) if (m := re.search(r"(\d+) skipped", proc.stdout)) else 0
    return {"returncode": proc.returncode, "passed": passed, "failed": failed,
            "errors": errors, "skipped": skipped, "tail": tail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--tests", action="store_true")
    args = ap.parse_args()
    base = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "HEAD"),
        "dirty_paths": [p for p in _git("status", "--porcelain").splitlines() if p.strip()],
        "models": models(),
        "kb_chunks": kb_chunks(),
        "datasets": datasets(),
        "services": services(),
    }
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if prev:
        for key in ("tests", "replay"):
            if key in prev:
                base[key] = prev[key]
    if args.tests:
        base["tests"] = tests()
    if args.replay:
        base["replay"] = asyncio.run(replay())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in base.items() if k != "replay"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

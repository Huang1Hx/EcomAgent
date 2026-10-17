"""Task 0 基线清单:字段固定、密钥只记存在性、评测集按实际行数计。"""
import json
from pathlib import Path


def test_baseline_manifest_shape():
    path = Path("data/eval/manifests/baseline.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {"captured_at", "git_sha", "dirty_paths", "models", "kb_chunks",
            "datasets", "services"} <= data.keys()
    assert data["datasets"]["tests/data/eval_ch04.jsonl"]["samples"] == 300
    for name, present in data["models"]["keys_present"].items():
        assert isinstance(present, bool), name


def test_baseline_manifest_has_no_secret_values():
    text = Path("data/eval/manifests/baseline.json").read_text(encoding="utf-8")
    for marker in ("sk-", "api_key=", "CHAT_API_KEY=", "Bearer "):
        assert marker not in text

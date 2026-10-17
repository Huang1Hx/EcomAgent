"""基线采集器只输出应存在性,不输出密钥值。"""
from scripts.eval import baseline


def test_models_reports_key_presence_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CHAT_MODEL=m\nCHAT_BASE_URL=http://x\nCHAT_API_KEY=sk-secret\n", encoding="utf-8")
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "_settings_value", lambda name: 0.5)
    out = baseline.models()
    assert out["keys_present"] == {"CHAT_API_KEY": True}
    assert "sk-secret" not in json_dump(out)
    assert out["chat_model"] == "m"


def json_dump(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)

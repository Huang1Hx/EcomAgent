"""数据版本、服务资产及训练选择规则的回归。"""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def artifacts():
    spec = importlib.util.find_spec("scripts.ch10.artifacts")
    assert spec is not None, "分类器缺少统一路径与产物校验模块"
    return importlib.import_module("scripts.ch10.artifacts")


def write_dataset(root):
    root.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        (root / f"{split}.jsonl").write_text(json.dumps(
            {"text": split, "labels": ["物流"]}, ensure_ascii=False), encoding="utf-8")


def test_dataset_default_independent_of_cwd(tmp_path, monkeypatch):
    a = artifacts()
    monkeypatch.chdir(tmp_path)
    assert a.dataset_manifest()["splits"]["train"]["count"] == 2594


def test_dataset_rejects_unknown_label(tmp_path):
    a = artifacts()
    write_dataset(tmp_path)
    (tmp_path / "train.jsonl").write_text('{"text":"x","labels":["unknown"]}')
    with pytest.raises(ValueError, match="label"):
        a.dataset_manifest(tmp_path)


def test_dataset_detects_cross_split_overlap(tmp_path):
    a = artifacts()
    write_dataset(tmp_path)
    (tmp_path / "val.jsonl").write_bytes((tmp_path / "train.jsonl").read_bytes())
    with pytest.raises(ValueError, match="overlap"):
        a.dataset_manifest(tmp_path)


def test_frozen_data_rejects_change(tmp_path):
    a = artifacts()
    write_dataset(tmp_path)
    frozen = a.dataset_manifest(tmp_path)
    (tmp_path / "test.jsonl").write_text('{"text":"changed","labels":["物流"]}')
    with pytest.raises(ValueError, match="dataset"):
        a.verify_dataset(frozen, tmp_path)


@pytest.mark.parametrize("changed", ["model.onnx", "tokenizer.json", "threshold.json"])
def test_service_rejects_corrupt_asset(tmp_path, changed):
    a = artifacts()
    for name in ("model.onnx", "tokenizer.json", "threshold.json"):
        (tmp_path / name).write_text(name)
    manifest = {"labels": list(a.TOPIC_NAMES), "files": a.file_hashes(tmp_path)}
    a.write_json(tmp_path / "artifact-manifest.json", manifest)
    a.verify_bundle(tmp_path)
    (tmp_path / changed).write_text("corrupted")
    with pytest.raises(ValueError, match="SHA256"):
        a.verify_bundle(tmp_path)


def test_service_rejects_label_order_change(tmp_path):
    a = artifacts()
    a.write_json(tmp_path / "artifact-manifest.json", {"labels": [], "files": {}})
    with pytest.raises(ValueError, match="labels"):
        a.verify_bundle(tmp_path)


def test_training_metrics_use_serving_fallback():
    pytest.importorskip("transformers")
    from scripts.ch10.train import compute_metrics
    result = compute_metrics((np.array([[-2., -3.]]), np.array([[1, 0]])))
    assert result["micro_f1"] == 1.0


@pytest.mark.parametrize("module", ["evaluate", "export_onnx", "scan_threshold_replay"])
def test_cli_can_select_dataset_without_loading_assets(module):
    if module != "scan_threshold_replay":
        pytest.importorskip("transformers")
    result = subprocess.run([sys.executable, "-m", f"scripts.ch10.{module}", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--data-dir" in result.stdout


def test_acceptance_page_reads_root_dataset(monkeypatch, tmp_path):
    from app.api import acceptance
    monkeypatch.chdir(tmp_path)
    assert acceptance._split_stats()["splits"]["train"]["size"] == 2594


def test_service_start_missing_bundle_returns_failure(tmp_path):
    a = artifacts()
    result = subprocess.run([sys.executable, "-m", "scripts.ch10.service", "start",
                             "--model-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "artifact" in result.stderr.lower()
    assert not (tmp_path / "classifier.pid").exists()


def test_running_service_with_old_threshold_is_rejected(tmp_path, monkeypatch):
    from scripts.ch10 import service
    a = artifacts()
    for name in ("model.onnx", "tokenizer.json", "threshold.json"):
        (tmp_path / name).write_text(name)
    manifest = {"labels": list(a.TOPIC_NAMES), "files": a.file_hashes(tmp_path),
                "dataset": a.dataset_manifest()}
    a.write_json(tmp_path / "artifact-manifest.json", manifest)
    monkeypatch.setattr(service, "health", lambda: {
        "onnx_sha256": manifest["files"]["model.onnx"], "bundle_sha256": "old-threshold"})
    with pytest.raises(RuntimeError, match="不同版本"):
        service.start(tmp_path)


def test_failed_export_preserves_previous_bundle_and_reports_failure(tmp_path):
    a = artifacts()
    assert hasattr(a, "publish_export"), "export must stage artifacts before publishing"
    old = tmp_path / "onnx"
    old.mkdir()
    (old / "model.onnx").write_bytes(b"valid-old-model")
    report = tmp_path / "export_report.json"
    a.write_json(report, {"passed": True})

    def fail_export(staging):
        (staging / "model.onnx").write_bytes(b"partial-new-model")
        raise ValueError("logits mismatch")

    with pytest.raises(ValueError, match="logits mismatch"):
        a.publish_export(fail_export, old, report)
    assert (old / "model.onnx").read_bytes() == b"valid-old-model"
    assert json.loads(report.read_text())["passed"] is False


def test_threshold_replay_rejects_wrong_running_bundle(monkeypatch):
    from scripts.ch10 import scan_threshold_replay as replay
    assert hasattr(replay, "verify_service"), "threshold replay must verify HTTP bundle identity"
    with pytest.raises(ValueError, match="bundle"):
        replay.verify_service({"bundle_sha256": "old"}, {"files": {"model.onnx": "new"}})

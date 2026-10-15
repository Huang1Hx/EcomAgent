"""数据版本、服务资产及训练选择规则的回归。"""
import importlib
import json

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

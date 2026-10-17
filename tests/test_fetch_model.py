"""Task 4 Step 4: 模型下载器——未授权/未配置时拒绝,下载后逐文件 SHA256 校验,失败不落正式目录。"""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.ch10 import fetch_model


def _bundle(tmp_path: Path) -> tuple[Path, dict, Path]:
    src = tmp_path / "src"
    src.mkdir()
    (src / "model.onnx").write_bytes(b"onnx-bytes")
    (src / "tokenizer.json").write_bytes(b"{}")
    (src / "threshold.json").write_bytes(b'{"threshold": 0.35}')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
    manifest = {"kind": "onnx", "files": files}
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in src.iterdir():
            z.write(p, p.name)
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest["artifact_uri"] = archive.resolve().as_uri()
    mp = tmp_path / "release.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    return archive, manifest, mp


def test_refuses_missing_manifest_and_unconfigured_release(tmp_path):
    assert fetch_model.main(["--manifest", str(tmp_path / "missing.json"),
                             "--out", str(tmp_path / "x"), "--check-only"]) == 1
    manifest = Path("data/ch10/artifact-manifest.json")
    assert json.loads(manifest.read_text(encoding="utf-8"))["artifact_uri"] is None
    with pytest.raises(fetch_model.FetchError):
        fetch_model.fetch(manifest, tmp_path / "x")


def test_bootstrap_refuses_without_artifact_uri(tmp_path):
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"artifact_uri": None, "files": {}}), encoding="utf-8")
    assert fetch_model.main(["--manifest", str(m), "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


def test_existing_valid_bundle_is_reused(tmp_path):
    out = tmp_path / "onnx"
    out.mkdir()
    (out / "model.onnx").write_bytes(b"onnx-bytes")
    (out / "tokenizer.json").write_bytes(b"{}")
    (out / "threshold.json").write_bytes(b'{"threshold": 0.35}')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    (out / "artifact-manifest.json").write_text(
        json.dumps({"kind": "onnx", "files": files, "labels": []}), encoding="utf-8")
    assert fetch_model.status(out) == "present"


def test_download_verifies_and_writes_then_reuses(tmp_path):
    _, _, manifest_path = _bundle(tmp_path)
    out = tmp_path / "onnx"
    assert fetch_model.fetch(manifest_path, out) == "downloaded"
    assert fetch_model.status(out) == "present"
    assert fetch_model.fetch(manifest_path, out) == "present"


def test_corrupt_archive_fails_and_keeps_target_absent(tmp_path):
    archive, _, manifest_path = _bundle(tmp_path)
    archive.write_bytes(b"not a zip")
    out = tmp_path / "onnx"
    with pytest.raises(fetch_model.FetchError):
        fetch_model.fetch(manifest_path, out)
    assert not out.exists()
    assert not list(out.parent.glob(".onnx-*"))


def test_wrong_file_hash_does_not_replace_target(tmp_path):
    archive, manifest, _ = _bundle(tmp_path)
    out = tmp_path / "onnx"
    out.mkdir()
    (out / "keep.txt").write_text("local", encoding="utf-8")
    manifest = dict(manifest, files={**manifest["files"], "model.onnx": "0" * 64})
    with pytest.raises(fetch_model.FetchError):
        fetch_model.install_from_archive(archive, manifest, out)
    assert (out / "keep.txt").read_text(encoding="utf-8") == "local"
    assert not (out / "model.onnx").exists()

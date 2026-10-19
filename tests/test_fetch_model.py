"""Task 4 Step 4: 模型下载器——未授权/未配置时拒绝,下载后逐文件 SHA256 校验,失败不落正式目录。"""
import hashlib
import io
import json
from email.message import Message
from urllib.error import HTTPError
import zipfile
from pathlib import Path
from types import SimpleNamespace

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
    manifest = tmp_path / "unconfigured.json"
    manifest.write_text(json.dumps({"artifact_uri": None, "files": {}}), encoding="utf-8")
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
    assert json.loads((out / "artifact-manifest.json").read_text())["artifact_uri"] is None
    assert fetch_model.fetch(manifest_path, out) == "present"


@pytest.mark.parametrize("changed_field,value", [
    ("kind", "torch"),
    ("dataset", {"version": "different-dataset"}),
])
def test_existing_bundle_with_metadata_drift_is_not_reused(tmp_path, changed_field, value):
    _, release, manifest_path = _bundle(tmp_path)
    out = tmp_path / "onnx"
    assert fetch_model.fetch(manifest_path, out) == "downloaded"
    local_path = out / "artifact-manifest.json"
    local = json.loads(local_path.read_text(encoding="utf-8"))
    local[changed_field] = value
    local_path.write_text(json.dumps(local), encoding="utf-8")
    assert fetch_model.status(out, release) == "invalid"
    with pytest.raises(fetch_model.FetchError, match="不同版本"):
        fetch_model.fetch(manifest_path, out)


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


def test_archive_rejects_directory_entries_even_when_empty(tmp_path):
    archive, manifest, _ = _bundle(tmp_path)
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("nested/", "")
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    out = tmp_path / "onnx"
    with pytest.raises(fetch_model.FetchError, match="目录"):
        fetch_model.install_from_archive(archive, manifest, out)
    assert not out.exists()


def test_private_github_release_download_uses_signed_url_without_token_leak(
        tmp_path, monkeypatch):
    uri = ("https://github.com/Huang1Hx/EcomAgent/releases/download/"
           "classifier-v1/model.zip")
    dest = tmp_path / "bundle.zip"
    seen = []

    api_asset = "https://api.github.com/repos/Huang1Hx/EcomAgent/releases/assets/123"
    signed = "https://release-assets.githubusercontent.com/model.zip?signature=temporary"

    def fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        if cmd[1:3] == ["api", "repos/Huang1Hx/EcomAgent/releases/tags/classifier-v1"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({
                "assets": [{"name": "model.zip", "url": api_asset}]}))
        assert cmd == ["gh", "auth", "token"]
        return SimpleNamespace(returncode=0, stdout="secret-token\n")

    class RedirectOpener:
        def open(self, request, timeout):
            assert request.full_url == api_asset
            assert request.get_header("Authorization") == "Bearer secret-token"
            headers = Message()
            headers["Location"] = signed
            raise HTTPError(api_asset, 302, "Found", headers, None)

    def fake_urlopen(request, timeout):
        assert request.full_url == signed
        assert request.get_header("Authorization") is None
        return io.BytesIO(b"release-asset")

    monkeypatch.setattr(fetch_model.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(fetch_model.subprocess, "run", fake_run)
    monkeypatch.setattr(fetch_model.urllib.request, "build_opener", lambda *a: RedirectOpener())
    monkeypatch.setattr(fetch_model.urllib.request, "urlopen", fake_urlopen)
    fetch_model._download(uri, dest)
    assert dest.read_bytes() == b"release-asset"
    assert len(seen) == 2
    assert all(kwargs["capture_output"] for _, kwargs in seen)
    assert "secret-token" not in " ".join(" ".join(cmd) for cmd, _ in seen)


def test_private_github_release_download_failure_hides_cli_output(tmp_path, monkeypatch):
    uri = ("https://github.com/Huang1Hx/EcomAgent/releases/download/"
           "classifier-v1/model.zip")
    monkeypatch.setattr(fetch_model.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(fetch_model.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stderr="secret-token-in-cli-output"))
    with pytest.raises(fetch_model.FetchError, match="gh auth login") as exc:
        fetch_model._download(uri, tmp_path / "bundle.zip")
    assert "secret-token-in-cli-output" not in str(exc.value)


def test_github_release_url_with_path_traversal_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_model.shutil, "which", lambda name: "/usr/bin/gh")
    with pytest.raises(fetch_model.FetchError):
        fetch_model._download(
            "https://github.com/Huang1Hx/EcomAgent/releases/download/"
            "classifier-v1/..%2Fsecret", tmp_path / "bundle.zip")

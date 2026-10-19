"""分类器模型远程分发:按发布清单下载 ONNX 包,逐文件 SHA256 校验后原子写入。

安全与授权边界:
- 只接受发布清单里已填写的 artifact_uri(https:// 或 file://)。私有 GitHub
  Release 链接由已登录的 gh CLI 下载,不把访问令牌写入清单或命令行。
- 下载先落临时文件,解压到同目录暂存区;校验 archive_sha256 与 manifest.files
  后整体替换目标目录。失败保留原目录,清理暂存,不产生半包。
- 只接受归档根目录下的扁平文件名,拒绝路径穿越、子目录和多余文件。

用法:
  PYTHONPATH=. uv run python -m scripts.ch10.fetch_model --mesh  # 预留拼写陷阱不计
  PYTHONPATH=. uv run python -m scripts.ch10.fetch_model --manifest data/ch10/artifact-manifest.json
  PYTHONPATH=. uv run python -m scripts.ch10.fetch_model --manifest ... --check-only
  make classifier-bootstrap
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/ch10/artifact-manifest.json"
OUT = ROOT / "data/ch10/onnx"
REQUIRED = {"model.onnx", "tokenizer.json", "threshold.json"}


class FetchError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _same(a: dict, b: dict) -> bool:
    return a.get("files") == b.get("files") and a.get("labels") == b.get("labels")


def status(out: Path = OUT, manifest: dict | None = None) -> str:
    """present=已有完整且哈希正确的本地产物;absent=目录为空或不存在;invalid=缺文件/哈希不符。"""
    out = Path(out)
    if not out.is_dir() or not any(out.iterdir()):
        return "absent"
    try:
        local = json.loads((out / "artifact-manifest.json").read_text(encoding="utf-8"))
        for name, digest in local["files"].items():
            if Path(name).name != name or not (out / name).is_file():
                return "invalid"
            if sha256(out / name) != digest:
                return "invalid"
        if manifest is not None and not _same(local, manifest):
            return "invalid"
        return "present"
    except (OSError, ValueError, KeyError, TypeError):
        return "invalid"


def _download(uri: str, dest: Path, timeout: float = 60) -> None:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme not in ("https", "file"):
        raise FetchError(f"不支持的 artifact_uri 协议:{parsed.scheme}(只允许 https/file)")
    if parsed.scheme == "file":
        src = Path(urllib.request.url2pathname(parsed.path))
        if not src.is_file():
            raise FetchError(f"本地 artifact 不存在:{src}")
        shutil.copyfile(src, dest)
        return
    if parsed.netloc == "github.com" and "/releases/download/" in parsed.path:
        _download_github_release(parsed, dest)
        return
    with urllib.request.urlopen(uri, timeout=timeout) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def _download_github_release(parsed: urllib.parse.ParseResult, dest: Path) -> None:
    """用 gh 当前登录态下载私有 Release 资产;绝不接收清单里的令牌。"""
    parts = [urllib.parse.unquote(part) for part in parsed.path.strip("/").split("/")]
    valid_part = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
    if (parsed.scheme != "https" or parsed.netloc != "github.com"
            or parsed.params or parsed.query or parsed.fragment
            or len(parts) != 6 or parts[2:4] != ["releases", "download"]
            or not all(valid_part.fullmatch(part) and part not in (".", "..")
                       for part in (*parts[:2], *parts[4:]))):
        raise FetchError("无效的 GitHub Release 资产地址")
    if not shutil.which("gh"):
        raise FetchError("私有 GitHub Release 下载需要安装 gh 并执行 gh auth login")
    owner, repo, _, _, tag, asset = parts
    cmd = ["gh", "release", "download", tag, "--repo", f"{owner}/{repo}",
           "--pattern", asset, "--dir", str(dest.parent)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise FetchError("GitHub Release 下载失败;请检查 gh auth login 与网络连接") from exc
    if result.returncode != 0:
        # gh 的错误输出可能包含终端环境信息,不要写入报告或异常消息。
        raise FetchError("GitHub Release 下载失败;请检查 gh auth login、仓库权限与网络连接")
    downloaded = dest.parent / asset
    if not downloaded.is_file():
        raise FetchError("GitHub Release 未返回清单指定的资产")
    downloaded.replace(dest)


def _safe_members(zf: zipfile.ZipFile) -> list[str]:
    names = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if "/" in name or "\\" in name or Path(name).name != name or name.startswith("."):
            raise FetchError(f"归档包含不安全路径:{name}")
        if info.file_size > 512 * 1024 * 1024:
            raise FetchError(f"归档文件超过 512MB:{name}")
        names.append(name)
    return names


def install_from_archive(archive: Path, manifest: dict, out: Path = OUT) -> None:
    """校验归档与逐文件哈希后原子替换目标目录;任何失败不触碰现有目录。"""
    if not isinstance(manifest, dict):
        raise FetchError("发布清单必须是 JSON 对象")
    expected_files = manifest.get("files") or {}
    missing = REQUIRED - set(expected_files)
    if missing:
        raise FetchError(f"发布清单缺少必需文件:{sorted(missing)}")
    if "artifact_uri" not in manifest or not manifest["artifact_uri"]:
        raise FetchError("发布清单 artifact_uri 为空:尚未配置已授权的远程存储目标")
    if not manifest.get("archive_sha256"):
        raise FetchError("发布清单缺少 archive_sha256,拒绝无校验下载")
    actual = sha256(archive)
    if actual != manifest["archive_sha256"]:
        raise FetchError(f"归档 SHA256 与发布清单不一致:{actual}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        with zipfile.ZipFile(archive) as zf:
            names = _safe_members(zf)
            extra = set(names) - set(expected_files) - {"artifact-manifest.json"}
            if extra:
                raise FetchError(f"归档含清单外文件:{sorted(extra)}")
            zf.extractall(staging)
        for name, digest in expected_files.items():
            path = staging / name
            if not path.is_file() or sha256(path) != digest:
                raise FetchError(f"文件 SHA256 不一致:{name}")
        local_manifest = dict(manifest)
        local_manifest.pop("archive_sha256", None)
        # 远端位置只属于发布清单,不属于模型身份;本地导出的原始包在此字段为 null。
        # 否则同一 ONNX 包在新环境下载后 bundle_id 会因 URL 不同而漂移。
        local_manifest["artifact_uri"] = None
        (staging / "artifact-manifest.json").write_text(
            json.dumps(local_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        backup = None
        if out.exists():
            backup = out.with_name(out.name + f".old-{os.getpid()}")
            out.rename(backup)
        try:
            staging.rename(out)
        except Exception:
            if backup is not None:
                backup.rename(out)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def fetch(manifest_path: Path = MANIFEST, out: Path = OUT) -> str:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FetchError(f"发布清单不存在:{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    uri = manifest.get("artifact_uri")
    if not uri:
        raise FetchError("发布清单 artifact_uri 为空:尚未配置已授权的远程存储目标;"
                         "请先取得存储授权并填写清单,本次不会从公开地址猜测下载。")
    current = status(out, manifest)
    if current == "present":
        return "present"
    if current == "invalid":
        raise FetchError(f"目标目录 {out} 已有不完整/不同版本的产物;"
                         "请人工移走后重试,下载器不会覆盖本地模型。")
    with tempfile.TemporaryDirectory(prefix="classifier-download-") as tmp:
        archive = Path(tmp) / "bundle.zip"
        _download(uri, archive)
        install_from_archive(archive, manifest, out)
    if status(out, manifest) != "present":
        raise FetchError("下载完成但校验未通过")
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check-only", action="store_true",
                    help="只检查本地产物是否完整;不下载")
    args = ap.parse_args(argv)
    if args.check_only:
        state = status(args.out)
        print(f"本地产物状态:{state}({args.out})")
        if state != "present":
            return 1
        return 0
    try:
        result = fetch(args.manifest, args.out)
    except SystemExit:
        raise
    except (FetchError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"model fetch 失败:{exc}", file=sys.stderr)
        return 2
    print({"present": "已存在且校验通过,跳过下载", "downloaded": "下载并校验完成"}[result])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

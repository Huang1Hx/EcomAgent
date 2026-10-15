"""启动/停止本地分类服务；校验失败或未就绪返回非零，避免伪成功。"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from scripts.ch10 import artifacts

PID = artifacts.ROOT / "data/classifier.pid"
LOG = artifacts.ROOT / "log/classifier.log"


def health():
    with urllib.request.urlopen("http://127.0.0.1:8110/healthz", timeout=1) as r:
        return json.load(r)


def start(model_dir):
    manifest = artifacts.verify_bundle(model_dir)
    artifacts.verify_dataset(manifest["dataset"])
    try:
        current = health()
    except (OSError, ValueError):
        current = None
    if current is not None:
        if current.get("onnx_sha256") == manifest["files"]["model.onnx"]:
            print("分类器已就绪: http://127.0.0.1:8110")
            return
        raise RuntimeError("8110 已有不同版本服务，请确认并停止旧服务后重试")
    # 不用 SO_REUSEADDR：有旧服务占用时先失败，不覆盖它的PID。
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8110))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("ab") as log:
        proc = subprocess.Popen([sys.executable, "-m", "scripts.ch10.serve"],
                                cwd=artifacts.ROOT,
                                env={**os.environ, "CLASSIFIER_MODEL_DIR": str(Path(model_dir).resolve())},
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    PID.write_text(str(proc.pid), encoding="utf-8")
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"分类器启动失败，查看 {LOG}")
            try:
                if health().get("onnx_sha256") == manifest["files"]["model.onnx"]:
                    print(f"分类器就绪: http://127.0.0.1:8110 (pid={proc.pid})")
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.25)
        raise RuntimeError(f"分类器就绪超时，查看 {LOG}")
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
        PID.unlink(missing_ok=True)
        raise


def stop():
    if not PID.exists():
        print("没有由本项目记录的分类器进程")
        return
    pid = int(PID.read_text())
    command = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True).stdout
    if not command.strip():
        PID.unlink()
        return
    if "-m scripts.ch10.serve" not in command:
        raise RuntimeError("PID不属于分类器，拒绝停止无关进程")
    os.kill(pid, signal.SIGTERM)
    PID.unlink()
    print("分类器已停止")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop"])
    ap.add_argument("--model-dir", type=Path, default=artifacts.ONNX)
    args = ap.parse_args()
    try:
        start(args.model_dir) if args.action == "start" else stop()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Classifier artifact/service error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

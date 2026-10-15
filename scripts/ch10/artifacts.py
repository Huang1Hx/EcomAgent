"""分类器数据版本和本地产物校验；只依赖标准库和项目术语表。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile
import uuid

from app.core.taxonomy import TOPIC_NAMES

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"
CH10 = ROOT / "data/ch10"
MODEL = CH10 / "model"
ONNX = CH10 / "onnx"
REPORTS = CH10 / "reports"


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def dataset_manifest(data_dir=DATA):
    data_dir = Path(data_dir).resolve()
    splits, texts = {}, {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_rows(path)
        if not rows:
            raise ValueError(f"empty dataset split: {split}")
        counts = Counter()
        seen = {}
        for row in rows:
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"invalid text: {split}")
            labels = row.get("labels")
            if (not isinstance(labels, list) or not labels
                    or any(lb not in TOPIC_NAMES for lb in labels)
                    or len(set(labels)) != len(labels)):
                raise ValueError(f"invalid labels: {split}")
            key = row["text"].strip()
            if key in seen and seen[key] != set(labels):
                raise ValueError(f"conflicting labels: {split}")
            seen[key] = set(labels)
            counts.update(labels)
        texts[split] = set(seen)
        splits[split] = {"count": len(rows), "sha256": sha256(path),
                         "duplicate_rows": len(rows) - len(seen),
                         "label_support": {lb: counts[lb] for lb in TOPIC_NAMES}}
    overlaps = {f"{a}_{b}": len(texts[a] & texts[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()):
        raise ValueError(f"dataset cross-split overlap: {overlaps}")
    digest = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    return {"version": "classifier-dataset-v1", "sha256": digest,
            "evaluation_scope": "existing_v1_regression_not_new_blind_test",
            "lineage": "original sample-level provenance unavailable",
            "labels": list(TOPIC_NAMES), "splits": splits, "overlaps": overlaps}


def verify_dataset(frozen, data_dir=DATA):
    current = dataset_manifest(data_dir)
    if current["sha256"] != frozen["sha256"] or current["labels"] != frozen["labels"]:
        raise ValueError("dataset version/SHA256 differs from training")
    return current


def file_hashes(directory):
    return {p.name: sha256(p) for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.name != "artifact-manifest.json" and not p.name.endswith(".tmp")}


def verify_bundle(directory=ONNX):
    directory = Path(directory)
    manifest = json.loads((directory / "artifact-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("labels") != list(TOPIC_NAMES):
        raise ValueError("artifact labels differ from current taxonomy")
    required = {"tokenizer.json", "threshold.json"}
    kind = manifest.get("kind", "torch" if "model.safetensors" in manifest.get("files", {}) else "onnx")
    required.add("model.safetensors" if kind == "torch" else "model.onnx")
    if not required <= manifest.get("files", {}).keys():
        raise ValueError("artifact manifest missing required files")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or not (directory / name).is_file():
            raise ValueError(f"artifact file missing or unsafe: {name}")
        if sha256(directory / name) != digest:
            raise ValueError(f"artifact SHA256 mismatch: {name}")
    return manifest


def bundle_id(manifest):
    """完整清单身份：权重相同但阈值、分词器或标签版本不同也不能复用。"""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def publish_export(build, out, report_path):
    """先生成并验证临时包，再替换正式目录；失败保留原包和失败现场。"""
    out, report_path = Path(out), Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="onnx-staging-", dir=out.parent))
    backup = None
    try:
        report = build(staging)
        if report.get("passed") is not True:
            raise ValueError("ONNX export validation failed")
        verify_bundle(staging)
        if out.exists():
            backup = out.with_name(f"onnx-backup-{uuid.uuid4().hex[:12]}")
            out.rename(backup)
        try:
            staging.rename(out)
        except BaseException:
            if backup is not None:
                backup.rename(out)
            raise
        report["onnx_path"] = str(out / "model.onnx")
        write_json(report_path, report)
        return report
    except Exception as exc:
        write_json(report_path, {"passed": False, "ran_at": datetime.now(timezone.utc).isoformat(),
                                 "error": f"{type(exc).__name__}: {exc}",
                                 "staging_path": str(staging),
                                 "previous_bundle": str(backup or out)})
        raise


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"ran_at": datetime.now(timezone.utc).isoformat(), "git_sha": git("rev-parse", "HEAD"),
            "dirty_paths": git("diff", "--name-only").splitlines(),
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                              for p in sorted((ROOT / "scripts/ch10").glob("*.py"))},
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "onnxruntime", "tokenizers", "scikit-learn")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--verify", type=Path)
    args = ap.parse_args()
    if args.verify:
        manifest = verify_bundle(args.verify)
        verify_dataset(manifest["dataset"], args.data_dir)
        print("Classifier artifacts verified")
    else:
        manifest = dataset_manifest(args.data_dir)
        write_json(CH10 / "dataset-manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))

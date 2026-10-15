"""ch10 ONNX 导出:torch 模型 → onnx(变长 batch/seq),导出后跑测试集校验与 torch 预测完全一致。
注意:torch 2.9+ 的 torch.onnx.export 默认 dynamo=True,导 HF 模型此处显式 dynamo=False
走 TorchScript 导出器 + dynamic_axes 稳定路线(Context7 查证)。运行:make ch10-export。
落 reports/export_report.json 给验收页读(/acceptance):校验条数、不一致条数、产物大小。"""
import datetime as dt
import json
import pathlib
import shutil
import argparse
import time

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scripts.ch10 import artifacts
from scripts.ch10.inference_lib import apply_threshold

MODEL_DIR = artifacts.MODEL
OUT = artifacts.ONNX
REPORTS = artifacts.REPORTS


def _write_report(checked: int, mismatch: int, metadata: dict) -> None:
    """一致性校验的结论落盘;失败也写,验收页要能看见「上次导出没过」而不是一片空白。"""
    onnx_path = OUT / "model.onnx"
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "export_report.json").write_text(json.dumps({
        **metadata,
        "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checked": checked, "mismatch": mismatch, "passed": mismatch == 0,
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size if onnx_path.exists() else 0,
        "opset": 17,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


class LogitsWrapper(torch.nn.Module):
    """HF 模型输出是 ModelOutput dict,导出前包一层只回 logits 张量。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=pathlib.Path, default=artifacts.DATA)
    cli = ap.parse_args()
    model_manifest = artifacts.verify_bundle(MODEL_DIR)
    dataset = artifacts.verify_dataset(model_manifest["dataset"], cli.data_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    wrapper = LogitsWrapper(model)
    sample = tokenizer(["买大了想退", "快递到哪了"], padding=True, return_tensors="pt")
    dyn = {0: "batch", 1: "seq"}
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(OUT / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": dyn, "attention_mask": dyn,
                      "token_type_ids": dyn, "logits": {0: "batch"}},
        dynamo=False,
        opset_version=17,
    )
    tokenizer.save_pretrained(OUT)          # 带出 tokenizer.json 给轻运行时用
    shutil.copy(MODEL_DIR / "threshold.json", OUT / "threshold.json")

    # 一致性校验:测试集全量,ONNX 与 torch 的过线标签必须完全一致。
    # 参照模型必须重新加载:tracing 会把 transformers v5 masking_utils 的分支
    # 打成常量、污染进程内已有模型(实测被污染的参照 logits 差到 1.88,而文件本身没问题)
    ref = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    ref.eval()
    threshold = json.loads((OUT / "threshold.json").read_text())["threshold"]
    texts = [r["text"] for r in artifacts.read_rows(cli.data_dir / "test.jsonl")]
    sess = ort.InferenceSession(str(OUT / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    mismatch = 0
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=128, return_tensors="pt")
            t_logits = ref(**enc).logits.numpy()
            (o_logits,) = sess.run(["logits"], {k: v.numpy() for k, v in enc.items()})
            assert np.allclose(t_logits, o_logits, atol=1e-3), "logits 超容差"
            t_pred = apply_threshold(1 / (1 + np.exp(-t_logits)), threshold)
            o_pred = apply_threshold(1 / (1 + np.exp(-o_logits)), threshold)
            mismatch += int((t_pred != o_pred).any(axis=1).sum())
    enc = tokenizer([texts[0]], truncation=True, padding=True, max_length=128, return_tensors="np")
    feed = {k: v for k, v in enc.items()}
    for _ in range(3):
        sess.run(["logits"], feed)
    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        sess.run(["logits"], feed)
        latencies.append((time.perf_counter() - start) * 1000)
    metadata = {**artifacts.provenance(), "dataset": dataset,
                "model_sha256": artifacts.sha256(MODEL_DIR / "model.safetensors"),
                "onnx_sha256": artifacts.sha256(OUT / "model.onnx"),
                "cpu_batch1_latency_ms": {"median": float(np.median(latencies)),
                                           "p95": float(np.percentile(latencies, 95)),
                                           "warmup": 3, "runs": 20},
                "peak_memory": "not_measured_in_isolated_runtime"}
    _write_report(len(texts), mismatch, metadata)
    if mismatch:
        raise SystemExit(f"ONNX 与 torch 预测不一致 {mismatch} 条,导出失败")
    manifest = {"kind": "onnx", "labels": list(artifacts.TOPIC_NAMES), "dataset": dataset,
                "model_sha256": metadata["model_sha256"], "files": artifacts.file_hashes(OUT),
                "training_git_sha": model_manifest["training"]["git_sha"],
                "artifact_uri": None}
    artifacts.write_json(OUT / "artifact-manifest.json", manifest)
    artifacts.write_json(artifacts.CH10 / "artifact-manifest.json", manifest)
    print(f"ONNX 导出并校验通过({len(texts)} 条预测完全一致):{OUT}/model.onnx")


if __name__ == "__main__":
    main()

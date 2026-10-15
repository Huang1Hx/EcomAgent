"""显式 RUN_CLASSIFIER_LIVE=1：真实ONNX HTTP + 隔离MySQL旁路验收。"""
import json
import os

import httpx
import pytest

from app.core.taxonomy import TOPIC_NAMES
from app.db import repository
from scripts.ch10 import artifacts, classify_pool

pytestmark = pytest.mark.skipif(os.environ.get("RUN_CLASSIFIER_LIVE") != "1",
                                reason="requires trained classifier at :8110")


async def test_http_all_v1_samples_match_local_onnx():
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from scripts.ch10.inference_lib import apply_threshold

    rows = artifacts.read_rows(artifacts.DATA / "test.jsonl")
    manifest = artifacts.verify_bundle()
    async with httpx.AsyncClient(timeout=120) as client:
        h = (await client.get("http://127.0.0.1:8110/healthz")).json()
        assert h["onnx_sha256"] == manifest["files"]["model.onnx"]
        assert h["dataset_sha256"] == manifest["dataset"]["sha256"]
        response = await client.post("http://127.0.0.1:8110/classify",
                                     json={"texts": [r["text"] for r in rows]})
        response.raise_for_status()
        results = response.json()["results"]
    tok = Tokenizer.from_file(str(artifacts.ONNX / "tokenizer.json"))
    tok.enable_truncation(max_length=128)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    sess = ort.InferenceSession(str(artifacts.ONNX / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    expected = []
    for start in range(0, len(rows), 32):
        enc = tok.encode_batch([r["text"] for r in rows[start:start + 32]])
        feed = {k: np.array([getattr(e, attr) for e in enc], dtype=np.int64)
                for k, attr in (("input_ids", "ids"), ("attention_mask", "attention_mask"),
                                ("token_type_ids", "type_ids"))}
        logits, = sess.run(["logits"], feed)
        pred = apply_threshold(1 / (1 + np.exp(-logits)), h["threshold"])
        expected.extend([[TOPIC_NAMES[i] for i, hit in enumerate(p) if hit] for p in pred])
    assert [r["labels"] for r in results] == expected


async def test_classify_pool_live_is_idempotent(db_session_factory, tmp_path, monkeypatch):
    # db_session_factory只指向独立测试库；不向业务库造数据，报告也在临时目录。
    monkeypatch.setattr(classify_pool, "REPORTS", tmp_path)
    rows = artifacts.read_rows(artifacts.DATA / "val.jsonl")[:3]
    for row in rows:
        qid = await repository.insert_low_confidence(None, row["text"], "retrieval_low_conf", None)
        rid = await repository.insert_review_item(row["text"], None)
        await repository.set_matched_review(qid, rid)
    await classify_pool.main(min_batch=10, force=True)
    first = json.loads((tmp_path / "classify_run.json").read_text())
    assert first["written"] == 3
    assert (await repository.topic_distribution())["total"] == 3
    assert await repository.list_unclassified_questions() == []
    await classify_pool.main(min_batch=10, force=True)
    second = json.loads((tmp_path / "classify_run.json").read_text())
    assert second["status"] == "empty"
    assert second["written"] == 0
    assert (await repository.topic_distribution())["total"] == 3

"""Task 6 发布评测聚合:只聚合、不重算;未运行/哈希不符/服务错误都不能算通过。"""
import json
from pathlib import Path

import pytest

from scripts import run_release_eval as rr


def _write(tmp_path: Path, name: str, data: dict, dataset: Path | None = None) -> Path:
    payload = dict(data)
    if dataset is not None:
        import hashlib
        payload.setdefault("meta", {})
        payload["meta"].setdefault("dataset_sha256",
                                    hashlib.sha256(dataset.read_bytes()).hexdigest())
    if "/" in name:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        sub = tmp_path / _SUBDIR[name]
        for parent in (*reversed(sub.parents[:-1]), sub):
            if not parent.exists():
                parent.mkdir()
        p = sub / name
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


_SUBDIR = {"agent_eval.json": "ch02/reports", "intent_eval.json": "ch06/reports",
           "rag_eval.json": "ch04/reports", "flywheel_closure_eval.json": "ch09/reports",
           "eval_report.json": "ch10/reports", "context_eval.json": "eval/reports"}


def test_missing_result_is_not_run_and_cannot_pass(tmp_path):
    stages = rr.collect(tmp_path, require_flywheel=True)
    assert stages["agent"]["status"] == "not_run"
    overall = rr.overall(stages)
    assert overall["status"] == "not_run"
    assert overall["passed"] is False


def test_provider_error_is_not_passed(tmp_path):
    _write(tmp_path, "intent_eval.json",
           {"meta": {}, "summary": {"correct": 0, "total": 0},
            "error": "APITimeoutError"})
    stages = rr.collect(tmp_path, require_flywheel=False)
    assert stages["intent"]["status"] == "provider_error"
    assert rr.overall(stages)["status"] in ("provider_error", "not_run")


def test_dataset_sha_mismatch_marks_stale(tmp_path):
    ds = tmp_path / "dataset.jsonl"
    ds.write_text('{"query": "a"}\n', encoding="utf-8")
    p = _write(tmp_path, "agent_eval.json",
               {"meta": {"dataset": str(ds), "dataset_sha256": "0" * 64},
                "summary": {"passed": 1, "total": 1, "all_pass": True}, "cases": []})
    stages = rr.collect(tmp_path, require_flywheel=False,
                        expected={ds: "1" * 64})
    assert stages["agent"]["status"] == "stale_report"
    assert not rr.overall(stages)["passed"]


def test_ci_uses_numerator_denominator():
    lo, hi = rr.wilson(50, 100)
    assert 0.40 < lo < 0.50 < hi < 0.60


def test_flywheel_requires_isolation_marker(tmp_path):
    p = _write(tmp_path, "flywheel_closure_eval.json",
               {"meta": {"source": "production"}, "summary": {"passed": 50, "total": 50},
                "cases": []})
    stages = rr.collect(tmp_path, require_flywheel=True)
    assert stages["flywheel"]["status"] == "invalid_report"


def test_evidence_maps_only_current_passed_reports(tmp_path):
    stages = rr.collect(tmp_path, require_flywheel=True)
    evidence = rr.project_evidence(stages)
    assert all(item["status"] in ("supported", "partially_supported", "unsupported")
               for item in evidence)
    assert not any(item["status"] == "supported" for item in evidence)


def test_flywheel_stale_or_writeback_problems_cannot_pass(tmp_path):
    import hashlib
    ds = tmp_path / "fw.jsonl"
    ds.write_text("{}\n", encoding="utf-8")
    data = {"meta": {"source": "isolated_eval", "dataset": str(ds),
                     "dataset_sha256": "0" * 64},
            "summary": {"passed": 50, "total": 50}, "cases": []}
    _write(tmp_path, "flywheel_closure_eval.json", data)
    stages = rr.collect(tmp_path, require_flywheel=True,
                        expected={ds: hashlib.sha256(ds.read_bytes()).hexdigest()})
    assert stages["flywheel"]["status"] == "stale_report"
    data["meta"]["dataset_sha256"] = hashlib.sha256(ds.read_bytes()).hexdigest()
    data["meta"]["writeback_problems"] = ["vectorized"]
    _write(tmp_path, "flywheel_closure_eval.json", data)
    stages = rr.collect(tmp_path, require_flywheel=True,
                        expected={ds: data["meta"]["dataset_sha256"]})
    assert stages["flywheel"]["status"] == "failed"


def test_classifier_aggregates_existing_v1_report(tmp_path, monkeypatch):
    """分类器报告用 test_size 记录样本数;不能因字段名不同就把 161 条回归标成未运行。"""
    from scripts import run_release_eval as rr

    files = {"onnx/model.onnx": "a" * 64}
    monkeypatch.setattr(rr, "_classifier_bundle", lambda: (files, "b" * 64, "c" * 64))
    ds = {"labels": ["x"] * 17, "evaluation_scope": "existing_v1_regression_not_new_blind_test"}
    (tmp_path / "ch10/reports").mkdir(parents=True)
    (tmp_path / "ch10/reports/eval_report.json").write_text(json.dumps({
        "model_sha256": "b" * 64, "test_size": 161, "dataset": ds,
        "micro": {"f1": 0.98}, "macro": {"f1": 0.97}}), encoding="utf-8")
    monkeypatch.setattr(rr, "ARTIFACT", tmp_path / "artifact-manifest.json")
    (tmp_path / "artifact-manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    stage = rr._classifier(tmp_path / "ch10/reports")
    assert stage["status"] == "passed"
    assert stage["metrics"]["n_samples"] == 161


def test_conformance_report_serializes_collected_dicts(tmp_path):
    """collect() 返回 Stage 字典;生成发布报告不能按属性读取,否则真实聚合直接崩。"""
    stages = rr.collect(tmp_path, require_flywheel=False)
    payload = rr._conformance_data(stages)
    assert payload["stages"]["agent"]["status"] == "not_run"
    text = rr._conformance_md(payload)
    assert "发布评测报告" in text and "not_run" in text


def test_rag_report_records_dataset_sha(tmp_path, monkeypatch):
    """RAG 报告必须带当前盲测集 SHA,否则聚合阶段无法判断报告是否过期。"""
    from scripts import eval_ch04

    ds = tmp_path / "rag.jsonl"
    ds.write_text('{"id": "x"}\n', encoding="utf-8")
    monkeypatch.setenv("RAG_EVAL_SET", str(ds))
    assert eval_ch04.report_dataset() == {
        "dataset": str(ds), "dataset_sha256": __import__("hashlib").sha256(
            ds.read_bytes()).hexdigest()}


def test_dataset_path_match_does_not_confuse_same_filename(tmp_path):
    """同名不同目录的数据集不能靠文件名相似匹配,否则会把过期报告当成当前数据。"""
    a = tmp_path / "a/agent_test.jsonl"
    b = tmp_path / "b/agent_test.jsonl"
    for p in (a, b):
        p.parent.mkdir()
        p.write_text("x\n", encoding="utf-8")
    bad = rr._dataset_check({"meta": {"dataset": str(a), "dataset_sha256": "old"}}, {b: "new"})
    assert bad == ""


def test_rag_evidence_requires_complete_generation_and_refusal(tmp_path):
    """只跑了检索段的 RAG 报告不能被映射为“检索+拒答已支持”。"""
    p = _write(tmp_path, "rag_eval.json",
               {"meta": {"dataset_role": "blind_test", "dataset_sha256": "a" * 64},
                "retrieval": {"hybrid_rerank": {"overall": {"recall": 0.9, "mrr": 0.6}}},
                "refusal": {"answerable": {}, "absent": {}}, "generation": None})
    stages = rr.collect(tmp_path, require_flywheel=False, expected={})
    assert stages["rag"]["status"] == "passed"
    ev = {e["capability"]: e["status"] for e in rr.project_evidence(stages)}
    assert ev["BGE-M3 + BM25 + RRF + BGE-Reranker"] == "supported"
    assert ev["四信号置信度与拒答治理"] == "partially_supported"


def test_rag_report_error_marks_service_unavailable(tmp_path):
    p = _write(tmp_path, "rag_eval.json", {"meta": {}, "error": "检索服务不可用"})
    assert rr.collect(tmp_path, require_flywheel=False, expected={})["rag"]["status"] == "service_unavailable"

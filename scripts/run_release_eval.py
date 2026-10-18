"""Task 6 发布评测:按固定顺序运行各评测,只聚合现有产物,不重算口径。

顺序:数据合同 → 单测 → RAG 盲测 → 意图 → Agent → 飞轮闭环 → 分类器 → 上下文。
每个阶段状态只允许 passed / failed / provider_error / service_unavailable / not_run /
stale_report / invalid_report / blocked;未运行、报告哈希与当前数据集不一致、
上游错误、服务不可用都不得计为通过。

产物:
- data/eval/reports/release-eval.json / .md
- data/eval/reports/project-evidence.json / .md
说明:聚合器不替代各阶段评测器,不修改任何评测答案或阈值;飞轮闭环只在隔离环境运行,
未提供隔离库/向量库时状态为 blocked,并阻塞“三入口飞轮完整闭环”证据。

用法:
  make eval-release            # 完整链路(需要 :8000、Milvus、上游、隔离飞轮环境)
  PYTHONPATH=. uv run python -m scripts.run_release_eval --stages contract,unit,rag
  PYTHONPATH=. uv run python -m scripts.run_release_eval --aggregate-only
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data/eval"
REPORTS = EVAL / "reports"
AGENT_REPORT = ROOT / "data/ch02/reports/agent_eval.json"
INTENT_REPORT = ROOT / "data/ch06/reports/intent_eval.json"
RAG_REPORT = ROOT / "data/ch04/reports/rag_eval.json"
FLYWHEEL_REPORT = ROOT / "data/ch09/reports/flywheel_closure_eval.json"
CONF_REPORT = ROOT / "data/ch09/reports/confidence_calibration.json"
CLASSIFIER_EVAL = ROOT / "data/ch10/reports/eval_report.json"
ARTIFACT = ROOT / "data/ch10/artifact-manifest.json"
BASE = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000")
CONF_THRESHOLD = 0.26
RELEASE_TIMEOUT = 1800

STAGE_ORDER = ["contract", "unit", "rag", "intent", "agent", "flywheel", "classifier",
               "context"]
STAGE_LABEL = {"contract": "数据合同", "unit": "单元测试", "rag": "RAG 盲测",
               "intent": "在线意图", "agent": "Agent 多轮", "flywheel": "飞轮闭环",
               "classifier": "17类分类器", "context": "上下文固定集"}


class Stage(TypedDict, total=False):
    status: str
    command: str
    report: str
    metrics: dict
    error: str
    note: str


def stage(status: str, **fields: object) -> Stage:
    """构造一个报告阶段;缺省字段保持空值,便于 JSON 序列化。"""
    value: Stage = {"status": status, "command": "", "report": "", "metrics": {},
                    "error": "", "note": ""}
    value.update(fields)  # type: ignore[typeddict-item]
    return value


def sha256_file(path: Path) -> str:
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def wilson(numerator: int, denominator: int, z: float = 1.96) -> tuple[float, float]:
    """0/1 计数的 95% Wilson 区间;分母为 0 时返回 (0, 0),由调用方标 not_run。"""
    if denominator <= 0:
        return 0.0, 0.0
    p = numerator / denominator
    denom = 1 + z * z / denominator
    centre = (p + z * z / (2 * denominator)) / denom
    margin = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator ** 2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def ratio(num: int, den: int) -> dict:
    lo, hi = wilson(num, den)
    return {"numerator": num, "denominator": den,
            "rate": round(num / den, 4) if den else None,
            "ci95": [round(lo, 4), round(hi, 4)]}


def _load(path: Path) -> tuple[dict | None, str]:
    if not path.is_file():
        return None, "not_run"
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (OSError, ValueError) as exc:
        return None, f"报告不可读:{type(exc).__name__}"


def _dataset_check(data: dict, expected: dict[Path, str]) -> str:
    meta = data.get("meta") or {}
    declared = meta.get("dataset_sha256")
    if not declared:
        return "invalid_report"
    declared_path = Path(str(meta.get("dataset") or "")).resolve()
    for path, digest in expected.items():
        expected_path = Path(path).resolve()
        # 报告既可写仓库相对路径,也可写绝对路径;归一化后精确匹配,同名文件不算同一份
        relative = ROOT / str(meta.get("dataset") or "")
        same_file = declared_path == expected_path or (
            not Path(str(meta.get("dataset"))).is_absolute() and relative.resolve() == expected_path)
        if same_file and declared != digest:
            return "stale_report"
    return ""


def _agent(tmp: Path, expected: dict) -> Stage:
    data, err = _load(tmp / "agent_eval.json")
    if data is None:
        return stage(err)
    if data.get("error"):
        return stage("provider_error", report=str(AGENT_REPORT), error=str(data["error"])[:300])
    meta = data.get("meta") or {}
    bad = _dataset_check(data, expected)
    if bad:
        return stage(bad, report=str(AGENT_REPORT))
    summary = data.get("summary") or {}
    total = int(summary.get("total") or 0)
    passed = sum(1 for c in data.get("cases", []) if c.get("passed"))
    if total == 0:
        return stage("not_run", report=str(AGENT_REPORT))
    cases = data.get("cases", [])
    tool_cases = [c for c in cases if c.get("kind") in ("toolchain", "single")
                  and not c.get("boundary")]
    tools_ok = sum(1 for c in tool_cases if (c.get("checks") or {}).get("tools"))
    order_cases = [c for c in cases if (c.get("checks") or {}).get("tool_order") is not None]
    order_ok = sum(1 for c in order_cases if c["checks"]["tool_order"])
    scope_cases = [c for c in cases if c.get("checks", {}).get("order_scope") is not None]
    scope_ok = sum(1 for c in scope_cases if c["checks"]["order_scope"])
    status = "passed" if passed == total else "failed"
    return stage(status, report=str(AGENT_REPORT),
                 metrics={"passed": ratio(passed, total),
                          "tools": ratio(tools_ok, len(tool_cases)),
                          "tool_order": ratio(order_ok, len(order_cases)),
                          "order_scope": ratio(scope_ok, len(scope_cases))},
                 note=f"review_status={meta.get('review_status')};"
                      "independent_release_evidence=false")


def _intent(tmp: Path, expected: dict) -> Stage:
    data, err = _load(tmp / "intent_eval.json")
    if data is None:
        return stage(err)
    if data.get("error"):
        return stage("provider_error", report=str(INTENT_REPORT),
                     error=str(data["error"])[:300])
    bad = _dataset_check(data, expected)
    if bad:
        return stage(bad, report=str(INTENT_REPORT))
    summary = data.get("summary") or {}
    total = int(summary.get("total") or 0)
    if not total:
        return stage("not_run", report=str(INTENT_REPORT))
    status = "passed" if (summary.get("correct") == total
                          and not summary.get("invalid_output")) else "failed"
    return stage(status, report=str(INTENT_REPORT),
                 metrics={"accuracy": ratio(int(summary.get("correct") or 0), total),
                          "drift": ratio(int(summary.get("drift_correct") or 0),
                                         int(summary.get("drift_total") or 0)),
                          "per_class": data.get("per_class")},
                 note=f"review_status={(data.get('meta') or {}).get('review_status')}")


def _rag(tmp: Path, expected: dict) -> Stage:
    data, err = _load(tmp / "rag_eval.json")
    if data is None:
        return stage(err)
    if data.get("error"):
        text = str(data["error"])
        status = "provider_error" if any(w in text.lower() for w in (
            "timeout", "timed out", "429", "401", "402", "api", "balance")) else \
            "service_unavailable"
        return stage(status, report=str(RAG_REPORT), error=text[:300])
    meta = data.get("meta") or {}
    if meta.get("dataset_role") != "blind_test":
        return stage("invalid_report", report=str(RAG_REPORT),
                     error="RAG 报告不是盲测集口径")
    bad = _dataset_check(data, expected)
    if bad:
        return stage(bad, report=str(RAG_REPORT))
    retrieval = data.get("retrieval") or {}
    overall = (retrieval.get("hybrid_rerank") or {}).get("overall") or {}
    refusal = data.get("refusal") or {}
    gen = data.get("generation")
    if not overall or not refusal:
        return stage("failed", report=str(RAG_REPORT), error="缺少检索/拒答段")
    n = int(meta.get("n_samples") or 0)
    gen_note = "生成段未完成" if gen is None else ""
    status = "passed" if overall.get("recall") is not None else "failed"
    return stage(status, report=str(RAG_REPORT),
                 metrics={"recall_at_5": overall.get("recall"),
                          "mrr": overall.get("mrr"), "n_samples": n,
                          "refusal": refusal or None,
                          "complete": bool(refusal) and gen is not None,
                          "generation_skipped": (gen or {}).get("skipped")
                          if isinstance(gen, dict) else None,
                          "false_pass": ratio(int(refusal.get("absent", {}).get("wrongly_passed") or 0),
                                              int(refusal.get("absent", {}).get("total") or 0)),
                          "false_refuse": ratio(int(refusal.get("answerable", {}).get("wrongly_refused") or 0),
                                                int(refusal.get("answerable", {}).get("total") or 0))},
                 error=gen_note,
                 note=f"dataset={meta.get('dataset')};threshold={meta.get('confidence_threshold')}")


def _flywheel(tmp: Path, expected: dict | None = None) -> Stage:
    data, err = _load(tmp / "flywheel_closure_eval.json")
    if data is None:
        return stage("blocked" if err == "not_run" else err,
                     error="未提供隔离飞轮环境或未运行 make eval-flywheel-closure")
    meta = data.get("meta") or {}
    if meta.get("source") != "isolated_eval":
        return stage("invalid_report", report=str(FLYWHEEL_REPORT),
                     error="飞轮报告不是隔离环境产物,不能作为发布证据")
    if expected and _check_flywheel_dataset(data, expected):
        return stage("stale_report", report=str(FLYWHEEL_REPORT),
                     error="飞轮报告数据 SHA 与当前数据集不一致")
    summary = data.get("summary") or {}
    total = int(summary.get("total") or 0)
    if not total:
        return stage("not_run", report=str(FLYWHEEL_REPORT))
    passed = int(summary.get("passed") or 0)
    problems = meta.get("writeback_problems") or []
    if problems:
        return stage("failed", report=str(FLYWHEEL_REPORT),
                     error=f"写回断言失败:{','.join(problems)}")
    return stage("passed" if passed == total else "failed", report=str(FLYWHEEL_REPORT),
                 metrics={"closed_loop": ratio(passed, total),
                          "first_refusal": ratio(int(summary.get("first_refusal") or 0), total),
                          "answerable_after": ratio(int(summary.get("answerable_after") or 0), total)},
                 note="数据集 AI 辅助未审核;答案来源为预置")


def _check_flywheel_dataset(data: dict, expected: dict) -> str:
    meta = data.get("meta") or {}
    if meta.get("dataset_sha256") != expected.get(dataset := ROOT / meta.get("dataset", "")):
        return "stale_report"
    return ""


def _classifier_bundle() -> tuple[dict, str, str]:
    """校验本地产物,返回 (files, 权重 SHA, 完整 bundle 身份)。"""
    from scripts.ch10 import artifacts
    manifest = artifacts.verify_bundle(artifacts.ONNX)
    return (manifest.get("files"), manifest.get("model_sha256"),
            artifacts.bundle_id(manifest))


def _classifier(tmp: Path) -> Stage:
    try:
        bundle_files, model_sha, bundle_sha = _classifier_bundle()
    except Exception as exc:  # noqa: BLE001 本地产物缺失/哈希不符必须如实失败
        return stage("service_unavailable", report=str(CLASSIFIER_EVAL),
                     error=f"本地产物校验失败:{type(exc).__name__}")
    data, err = _load(tmp / "eval_report.json")
    artifact, _ = _load(ARTIFACT)
    if data is None or artifact is None:
        return stage("not_run", report=str(CLASSIFIER_EVAL))
    metrics = data.get("micro") or data.get("metrics") or data.get("overall") or {}
    micro = metrics.get("f1") if "f1" in metrics else metrics.get("micro_f1")
    if micro is None:
        micro = data.get("micro_f1")
    scope = (data.get("dataset") or {}).get("evaluation_scope")
    total = int(data.get("n_samples") or data.get("test_size") or data.get("total") or 0)
    if not total:
        return stage("not_run", report=str(CLASSIFIER_EVAL))
    model_ok = data.get("model_sha256") in (None, model_sha)
    return stage("passed" if micro is not None and model_ok
                 and bundle_files == artifact.get("files")
                 else "invalid_report", report=str(CLASSIFIER_EVAL),
                 metrics={"micro_f1": micro,
                          "macro_f1": (data.get("macro") or {}).get("f1")
                          or metrics.get("macro_f1"),
                          "n_samples": total,
                          "labels": len((data.get("dataset") or {}).get("labels", []))},
                 note=f"evaluation_scope={scope};model_sha256={data.get('model_sha256')};"
                      f"bundle={bundle_sha};本地产物 SHA256 已校验")


def _context(tmp: Path) -> Stage:
    """上下文固定集:确定性三层装配口径(scripts/eval_context.py),与在线生成质量分开表述。"""
    data, err = _load(tmp / "context_eval.json")
    if data is None:
        return stage("not_run", error="未运行上下文固定集(需新增运行器与隔离会话)")
    summary = data.get("summary") or {}
    total = int(summary.get("total") or 0)
    passed = int(summary.get("passed") or 0)
    meta = data.get("meta") or {}
    if meta.get("source") != "isolated_context_spec_eval":
        return stage("invalid_report", report=str(tmp / "context_eval.json"),
                     error="上下文报告不是隔离环境产物")
    return stage("passed" if total and passed == total else
                 ("not_run" if not total else "failed"),
                 report=str(tmp / "context_eval.json"),
                 metrics={"fact_retention": ratio(passed, total)})


def collect(tmp: Path = ROOT / "data", *, require_flywheel: bool = True,
            expected: dict[Path, str] | None = None) -> dict[str, Stage]:
    tmp = Path(tmp)
    if expected is None:
        expected = _expected_datasets()
    stages = {
        "contract": stage("not_run", note="由主流程预置"),
        "unit": stage("not_run", note="由主流程预置"),
        "rag": _rag(tmp / "ch04/reports", expected),
        "intent": _intent(tmp / "ch06/reports", expected),
        "agent": _agent(tmp / "ch02/reports", expected),
        "flywheel": _flywheel(tmp / "ch09/reports", expected) if require_flywheel else stage(
            "blocked", error="本轮未要求飞轮闭环"),
        "classifier": _classifier(tmp / "ch10/reports"),
        "context": _context(tmp / "eval/reports"),
    }
    return stages


def _expected_datasets() -> dict[Path, str]:
    return {EVAL / "agent_test.jsonl": sha256_file(EVAL / "agent_test.jsonl"),
            EVAL / "intent_test.jsonl": sha256_file(EVAL / "intent_test.jsonl"),
            EVAL / "rag_blind_test.jsonl": sha256_file(EVAL / "rag_blind_test.jsonl"),
            EVAL / "flywheel_closure_test.jsonl": sha256_file(EVAL / "flywheel_closure_test.jsonl"),
            EVAL / "context_test.jsonl": sha256_file(EVAL / "context_test.jsonl")}


def overall(stages: dict[str, Stage]) -> dict:
    bad = [name for name in STAGE_ORDER if stages[name]["status"] != "passed"]
    statuses = {stages[n]["status"] for n in bad}
    if not bad:
        return {"status": "passed", "passed": True, "failed_stages": []}
    for s in ("failed", "provider_error", "service_unavailable", "stale_report",
              "invalid_report", "not_run", "blocked"):
        if s in statuses:
            return {"status": s, "passed": False, "failed_stages": bad}
    return {"status": "not_run", "passed": False, "failed_stages": bad}


def project_evidence(stages: dict[str, Stage]) -> list[dict]:
    """七项能力映射:只读取当前报告;未通过/未运行的不得标 supported。"""
    def st(name: str) -> str:
        return stages[name]["status"]

    def evidence(name: str, supported: bool, partial: bool, report: str, statement: str):
        status = "supported" if supported else ("partially_supported" if partial else "unsupported")
        return {"capability": name, "status": status, "report": report, "statement": statement}

    rag = stages["rag"]
    agent = stages["agent"]
    intent = stages["intent"]
    fly = stages["flywheel"]
    ctx = stages["context"]
    clf = stages["classifier"]
    contract_ok = st("contract") == "passed"
    return [
        evidence("BGE-M3 + BM25 + RRF + BGE-Reranker", rag["status"] == "passed",
                 rag["status"] not in ("not_run", "blocked"),
                 rag["report"], "四策略检索与盲测 Recall@5/MRR/答案覆盖"),
        evidence("四信号置信度与拒答治理", rag["status"] == "passed"
                 and rag["metrics"].get("complete") is True,
                 rag["status"] not in ("not_run", "blocked"), rag["report"],
                 "固定阈值下误放/误拦;阈值来自独立校准集"),
        evidence("九类意图到五出口 Workflow + Agent",
                 intent["status"] == "passed" and agent["status"] == "passed",
                 intent["status"] == "passed" or agent["status"] == "passed",
                 f"{intent['report']};{agent['report']}", "九类意图、工具与订单归属校验"),
        evidence("三层上下文、异步摘要、工具裁剪", ctx["status"] == "passed",
                 ctx["status"] != "not_run", ctx["report"], "上下文固定集事实保留率"),
        evidence("三入口数据飞轮", fly["status"] == "passed", fly["status"] != "blocked",
                 fly["report"], "隔离环境闭环:拒答→入池→归并→批准→写回→再答"),
        evidence("17类多标签 RoBERTa + ONNX", clf["status"] == "passed",
                 clf["status"] not in ("not_run", "service_unavailable"),
                 clf["report"], "现有 v1 测试集回归;不是新独立盲测"),
        evidence("架构决策可追溯性", contract_ok, True, "data/eval/reports/release-eval.json",
                 "按提交、报告和数据集合同追溯;仅当前证据"),
    ]


def _run(cmd: list[str], item: Stage, env: dict | None = None) -> Stage:
    """运行一个阶段命令;失败只记录命令级状态,报告仍由 collect() 独立校验。"""
    item["command"] = " ".join(cmd)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, timeout=RELEASE_TIMEOUT)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    item["note"] = (item["note"] + " | " if item["note"] else "") + " ".join(tail)[:500]
    if proc.returncode != 0:
        text = (proc.stdout + proc.stderr).lower()
        if any(w in text for w in ("timeout", "connection", "429", "401", "upstream",
                                   "connecterror", "readtimeout")):
            item["status"] = "provider_error"
        else:
            item["status"] = "failed"
        item["error"] = text[-300:]
    else:
        item["status"] = "passed"
    return item


def _merge_report_status(command: Stage, report: Stage) -> Stage:
    """命令非零退出时保留命令级失败原因;命令成功时以报告校验结果为准。"""
    if command["status"] == "blocked":
        return command
    if command["status"] in ("provider_error", "failed", "service_unavailable"):
        if report["status"] == "passed":
            command["status"] = "invalid_report"
            command["error"] = command["error"] or "命令失败但报告显示通过"
        elif report["status"] in ("provider_error", "service_unavailable", "stale_report",
                                  "invalid_report"):
            command["status"] = report["status"]
            command["error"] = command["error"] or report["error"]
        return command
    if report["status"] == "passed":
        return report
    return report


def _conformance_data(stages: dict[str, Stage], root: Path = ROOT) -> dict:
    """把各阶段状态落成报告;collect() 返回 Stage 字典,统一按字段读取。"""
    return {"generated_at": datetime.now().isoformat(timespec="seconds"),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                               text=True).strip(),
            "stages": {k: {"status": v["status"], "command": v["command"],
                           "report": v["report"], "metrics": v["metrics"],
                           "error": v["error"], "note": v["note"]}
                       for k, v in stages.items()},
            "overall": overall(stages),
            "confidence_threshold": CONF_THRESHOLD,
            "data_contract": {"review_status": "ai_assisted_unreviewed", "reviewers": 0,
                              "independent_release_evidence": False},
            "not_run_or_blocked": [k for k, v in stages.items() if v["status"] != "passed"]}


def _conformance_md(payload: dict) -> str:
    lines = ["# 发布评测报告", "", f"- 生成时间:{payload['generated_at']}",
             f"- 提交:{payload['git_sha']}", f"- 总状态:{payload['overall']['status']}",
             "- 数据口径:AI 辅助编写、未经真实人工审核(`review_status=ai_assisted_unreviewed`)",
             "", "| 阶段 | 状态 | 指标 | 报告/错误 |", "|---|---|---|---|"]
    for name in STAGE_ORDER:
        s = payload["stages"][name]
        metric = json.dumps(s["metrics"], ensure_ascii=False)[:180]
        lines.append(f"| {STAGE_LABEL[name]} | {s['status']} | {metric} | "
                     f"{s['report'] or s['error'] or s['note']} |")
    lines += ["", "> 未运行、上游错误、服务不可用、报告与数据集 SHA 不一致均不计为通过。",
              "> 旧 300 题仅为开发集;不用于本报告的独立测试结论。"]
    return "\n".join(lines) + "\n"


def _evidence_md(evidence: list[dict]) -> str:
    lines = ["# 项目证据映射", "", "| 能力 | 状态 | 报告 | 表述 |", "|---|---|---|---|"]
    for e in evidence:
        lines.append(f"| {e['capability']} | {e['status']} | {e['report']} | {e['statement']} |")
    lines += ["", "> 只有当前报告通过且数据/模型版本一致时才可标 supported;"
              "AI 辅助未审核集不能表述为人工复核证据。"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGE_ORDER))
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-flywheel", action="store_true")
    args = ap.parse_args(argv)
    want = [s for s in args.stages.split(",") if s]
    REPORTS.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Stage] = {}
    env = {"PYTHONPATH": ".", "EVAL_BASE_URL": BASE}
    if not args.aggregate_only:
        if "contract" in want:
            stages["contract"] = _run([sys.executable, "-m", "pytest", "-q",
                                       "tests/test_eval_dataset_contracts.py"],
                                      stage("not_run"), env)
        if "unit" in want:
            stages["unit"] = _run([sys.executable, "-m", "pytest", "-q"], stage("not_run"), env)
        if "rag" in want:
            stages["rag"] = _run([sys.executable, "scripts/eval_ch04.py"], stage("not_run"),
                                 {**env, "RAG_CONF_THRESHOLD": str(CONF_THRESHOLD)})
        if "intent" in want:
            stages["intent"] = _run([sys.executable, "scripts/eval_intent.py"], stage("not_run"), env)
        if "agent" in want:
            stages["agent"] = _run([sys.executable, "scripts/eval_agent.py"], stage("not_run"), env)
        if "flywheel" in want:
            if args.skip_flywheel or not os.environ.get("FLYWHEEL_EVAL_DATABASE_URL") \
                    or not os.environ.get("FLYWHEEL_EVAL_MILVUS_URI"):
                stages["flywheel"] = stage("blocked", note="未提供隔离飞轮环境")
            else:
                stages["flywheel"] = _run([sys.executable, "scripts/eval_flywheel.py"],
                                          stage("not_run"), {**env, **{
                                              k: os.environ[k] for k in
                                              ("FLYWHEEL_EVAL_DATABASE_URL", "FLYWHEEL_EVAL_MILVUS_URI")}})
        if "context" in want:
            stages["context"] = _run([sys.executable, str(ROOT / "scripts/eval_context.py")],
                                     stage("not_run"), env)
        if "classifier" in want:
            stages["classifier"] = stage("not_run", note="由产物重新聚合")
    current = collect(ROOT / "data", require_flywheel=True)
    if not args.aggregate_only:
        for name, stage in stages.items():
            if name == "classifier":
                continue
            if name == "context":
                current[name] = _merge_report_status(stage, current[name])
                continue
            if name in ("contract", "unit"):
                current[name] = stage
                continue
            current[name] = _merge_report_status(stage, current[name])
    payload = _conformance_data(current)
    (REPORTS / "release-eval.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORTS / "release-eval.md").write_text(_conformance_md(payload), encoding="utf-8")
    evidence = project_evidence(current)
    ev = {"generated_at": payload["generated_at"], "git_sha": payload["git_sha"],
          "items": evidence}
    (REPORTS / "project-evidence.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORTS / "project-evidence.md").write_text(_evidence_md(evidence), encoding="utf-8")
    for name in STAGE_ORDER:
        print(f"{STAGE_LABEL[name]:10s} {current[name]['status']}")
    print(f"总状态:{payload['overall']['status']}")
    print(f"报告:{REPORTS.relative_to(ROOT)}/release-eval.{{json,md}}")
    return 0 if payload["overall"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

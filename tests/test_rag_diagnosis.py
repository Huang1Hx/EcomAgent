"""N3 离线诊断:只读题集与已存证据,区分「解释性拒答」与「错误承诺」。"""
import json


def test_explanatory_refusal_requires_evidence_without_commitment():
    from scripts.rag_diagnosis import classify_absent
    kb = [{"section_path": "退款规则/到账时效",
           "answer": "平台不承诺具体到账天数,不要向用户给出「几个工作日到账」这类具体数字。"}]
    assert classify_absent({"id": "x"}, kb, sufficient=True)["legacy_outcome"] == "false_pass"
    verdict = classify_absent({"id": "x"}, kb, sufficient=True)
    assert verdict["revised_outcome"] == "explanatory_refusal_ok"
    assert classify_absent({"id": "x"}, [], sufficient=True)["revised_outcome"] == "unsupported_false_pass"
    assert classify_absent({"id": "x"}, kb, sufficient=False)["revised_outcome"] == "refused"


def test_commitment_is_checked_separately_from_evidence():
    from scripts.rag_diagnosis import commitment_in_answer
    assert commitment_in_answer("退款会在 3 个工作日内到账") is True
    assert commitment_in_answer("平台不承诺具体到账天数,以支付渠道为准") is False
    from scripts.rag_diagnosis import classify_absent
    kb = [{"section_path": "退款规则/到账时效", "answer": "平台不承诺具体到账天数。"}]
    verdict = classify_absent({"id": "x"}, kb, sufficient=True, answer="3 个工作日内到账")
    assert verdict["revised_outcome"] == "false_commitment"


def test_bucket_audit_reports_anchor_coverage_and_duplicates():
    from scripts.rag_diagnosis import bucket_audit
    rows = [
        {"id": "a", "bucket": "D_absent", "should_refuse": True, "query": "q1"},
        {"id": "b", "bucket": "A_policy", "should_refuse": False, "query": "q1",
         "expect_section": "s", "expect_points": ["p"]},
        {"id": "c", "bucket": "A_policy", "should_refuse": False, "query": "q2",
         "expect_sections_all": [["s"]]},
    ]
    audit = bucket_audit(rows)
    assert audit["n"] == 3
    assert audit["buckets"]["D_absent"] == {"n": 1, "refuse": 1, "anchored": 0}
    assert audit["buckets"]["A_policy"]["anchored"] == 2
    assert audit["duplicate_queries"] == {"q1": ["a", "b"]}
    assert audit["missing_points"] == ["c"]


def test_retrieval_summary_keeps_numerator_and_denominator():
    from scripts.rag_diagnosis import retrieval_summary
    rows = [{"id": "a"}, {"id": "b"}]
    hits = {"a": [{"section_path": "s"}], "b": []}
    out = retrieval_summary(rows, hits, expect_by_id={"a": "s", "b": "s"})
    assert out == {"n": 2, "recall_at_5_hits": 1, "recall_at_5": 0.5,
                   "mrr_numerator": 1.0, "mrr_denominator": 2, "mrr": 0.5}


def test_saved_report_denominators_match_samples():
    import subprocess

    from scripts.rag_diagnosis import check_saved_report
    rows = [json.loads(l) for l in open("data/eval/rag_blind_test.jsonl") if l.strip()]
    # 工作区的 rag_eval.json 可能正被小样/中断运行覆盖;此测试锁的是已提交的全量历史口径
    saved = subprocess.run(["git", "show", "HEAD:data/ch04/reports/rag_eval.json"],
                           capture_output=True, text=True, check=True).stdout
    report = json.loads(saved)
    out = check_saved_report(rows, report)
    assert out["answerable"]["report"] == 282
    assert out["absent"]["report"] == 114
    assert out["n"] == 396
    assert out["ok"] is True

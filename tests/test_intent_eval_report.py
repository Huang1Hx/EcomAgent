"""意图报告的分母必须与题集真实标注一致,未标注边界题不能冒充判错或判对。"""
from scripts import eval_intent


def test_unlabeled_boundary_cases_are_not_part_of_accuracy_denominator():
    payload = eval_intent.build_report(
        [{"id": "L1", "kind": "single", "query": "q", "expected": "物流",
          "actual": "物流", "confidence": 0.9, "valid_output": True, "correct": True},
         {"id": "B1", "kind": "boundary", "query": "b", "expected": None,
          "actual": "其他", "confidence": 0.8, "valid_output": True, "correct": False}],
        dataset_sha256="x", model="m")
    assert payload["summary"]["correct"] == 1
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["unlabeled"] == 1
    assert payload["summary"]["boundary_other"] == 1
    assert payload["summary"]["boundary_total"] == 1
    assert payload["summary"]["boundary_labeled"] == 0
    assert payload["summary"]["boundary_role"] == "unlabeled_probe"

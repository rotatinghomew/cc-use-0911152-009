from __future__ import annotations

import unittest

from learning_evidence.programs.contract import DomainRecord, stable_hash
from learning_evidence.analytics.missingness import (
    STRATEGY_ORDER,
    build_review,
    conclusion_support,
    consent_status,
    diagnose,
    missing_pattern,
    sensitivity,
    supportable_scope,
)


def make(rid: str, missing: tuple[str, ...] = (), **overrides) -> DomainRecord:
    payload = {
        "school_code": rid,
        "grade_band": "g5",
        "subject": "math",
        "learner_count": 120.0,
        "usage_minutes": 900.0,
        "baseline_score": 60.0,
        "outcome_score": 66.0,
        "missing_rate": 0.1,
        "teacher_support": 0.5,
        "connectivity": 0.8,
        "language_group": "lang-a",
        "consent_rate": 0.9,
        "region": "north",
    }
    for field in missing:
        payload.pop(field, None)
    payload.update(overrides)
    return DomainRecord(rid, payload)


class MissingPatternTests(unittest.TestCase):
    def test_pattern_classification(self):
        self.assertEqual(missing_pattern(make("a"))["pattern"], "complete")
        self.assertEqual(missing_pattern(make("b", ("outcome_score",)))["pattern"], "missing_outcome")
        self.assertEqual(missing_pattern(make("c", ("baseline_score", "outcome_score")))["pattern"],
                         "missing_outcome")
        self.assertEqual(missing_pattern(make("d", ("usage_minutes",)))["pattern"], "missing_usage")
        self.assertEqual(
            missing_pattern(make("e", ("baseline_score", "outcome_score", "usage_minutes")))["pattern"],
            "missing_outcome_and_usage")
        self.assertEqual(missing_pattern(make("f", ("teacher_support",)))["pattern"], "missing_other")

    def test_consent_banding(self):
        self.assertEqual(consent_status(make("a", consent_rate=0.95), {}), "full")
        self.assertEqual(consent_status(make("b", consent_rate=0.6), {}), "partial")
        self.assertEqual(consent_status(make("c", consent_rate=0.3), {}), "limited")
        self.assertEqual(consent_status(make("d", ("consent_rate",)), {}), "unknown")


class DiagnosticsTests(unittest.TestCase):
    def test_missingness_associated_with_region_and_consent(self):
        records = [
            make("n1"),
            make("n2", consent_rate=0.85),
            make("e1", ("baseline_score", "outcome_score"), region="east", consent_rate=0.4),
            make("e2", ("usage_minutes",), region="east", consent_rate=0.3),
        ]
        result = diagnose(records)
        self.assertEqual(result["pattern_counts"]["missing_outcome"], 1)
        self.assertEqual(result["pattern_counts"]["missing_usage"], 1)
        self.assertTrue(result["dimensions"]["region"]["associated"])
        self.assertTrue(result["dimensions"]["consent_scope"]["associated"])
        self.assertFalse(result["region_is_proxy"])
        self.assertEqual(result["missing_outcome_schools"], ["e1"])
        self.assertEqual(result["missing_usage_schools"], ["e2"])

    def test_uniform_missingness_is_not_flagged(self):
        records = [make("a"), make("b"), make("c"), make("d")]
        result = diagnose(records)
        for dimension in result["dimensions"].values():
            self.assertFalse(dimension["associated"])

    def test_region_falls_back_to_language_group_proxy(self):
        def without_region(rid: str, language: str) -> DomainRecord:
            payload = make(rid, language_group=language).payload
            return DomainRecord(rid, {k: v for k, v in payload.items() if k != "region"})
        records = [without_region("a", "lang-a"), without_region("b", "lang-b")]
        result = diagnose(records)
        self.assertTrue(result["region_is_proxy"])
        self.assertEqual(result["region_field"], "language_group")


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        # s3 只提供极端观测值（自身缺使用记录），worst_case 据此把缺失成绩填为不利极值。
        self.records = [
            make("s1"),
            make("s2", ("baseline_score", "outcome_score")),
            make("s3", ("usage_minutes",), baseline_score=100.0, outcome_score=1.0),
        ]

    def test_strategies_cover_named_assumptions(self):
        result = sensitivity(self.records)
        self.assertEqual(tuple(result["strategies"]), STRATEGY_ORDER)
        complete = result["strategies"]["complete_case"]
        self.assertEqual(complete["included_schools"], ["s1"])
        self.assertEqual(result["strategies"]["published_zero_fill"]["included_count"], 3)
        self.assertEqual(result["published_basis"], "published_zero_fill")

    def test_bounds_bracket_estimates_and_nothing_is_auto_selected(self):
        result = sensitivity(self.records)
        best = result["strategies"]["best_case"]["effectiveness"]
        worst = result["strategies"]["worst_case"]["effectiveness"]
        self.assertGreaterEqual(best, worst)  # 填充方向：最有利不低于最不利
        values = [row["effectiveness"] for row in result["strategies"].values() if row["defined"]]
        bounds = result["effectiveness_range"]
        self.assertAlmostEqual(bounds["min"], min(values))
        self.assertAlmostEqual(bounds["max"], max(values))
        self.assertIsNone(result["selection"])
        self.assertIn("未自动采用", result["selection_note"])

    def test_conclusion_marks_assumption_dependence(self):
        result = sensitivity(self.records)
        support = conclusion_support(result)
        positive = support["conclusions"]["positive_effect"]
        self.assertEqual(positive["status"], "assumption_dependent")
        self.assertIn("worst_case", positive["fails_under"])
        self.assertIn("complete_case", positive["holds_under"])
        self.assertEqual(positive["observations"], ["s1"])
        fair = support["conclusions"]["fair_across_groups"]
        self.assertEqual(fair["status"], "observation_supported")
        self.assertFalse(fair["depends_on_assumptions"])

    def test_conclusion_unsupported_when_no_strategy_holds(self):
        records = [make("a", outcome_score=40.0), make("b", ("usage_minutes",), outcome_score=35.0)]
        support = conclusion_support(sensitivity(records))
        self.assertEqual(support["conclusions"]["positive_effect"]["status"], "unsupported")


class SupportableScopeTests(unittest.TestCase):
    def test_group_without_complete_observation_is_not_comparable(self):
        records = [
            make("a1", language_group="lang-a"),
            make("b1", ("baseline_score", "outcome_score"), language_group="lang-b"),
            make("b2", ("usage_minutes",), language_group="lang-b"),
        ]
        scope = supportable_scope(records)
        self.assertEqual(scope["not_comparable_groups"], ["lang-b"])
        self.assertEqual(scope["supportable_schools"], ["a1"])
        self.assertEqual(scope["schools"]["b1"]["reasons"], ["group_not_comparable"])

    def test_consent_scope_and_unknown_consent_exclude_schools(self):
        records = [
            make("a1", language_group="lang-a"),
            make("a2", consent_rate=0.3, language_group="lang-a"),
            make("a3", ("consent_rate",), language_group="lang-a"),
        ]
        scope = supportable_scope(records)
        self.assertEqual(scope["supportable_schools"], ["a1"])
        self.assertEqual(scope["schools"]["a2"]["reasons"], ["consent_out_of_scope"])
        self.assertEqual(scope["schools"]["a3"]["reasons"], ["consent_unknown"])

    def test_expected_school_absent_from_warehouse_is_flagged(self):
        scope = supportable_scope([make("a1")], expected_schools=["a1", "ghost"])
        self.assertEqual(scope["schools"]["ghost"]["reasons"], ["record_missing"])
        self.assertEqual(scope["records_missing_from_warehouse"], ["ghost"])

    def test_minimum_group_complete_is_configurable(self):
        records = [make("a1", language_group="lang-a"),
                   make("a2", ("usage_minutes",), language_group="lang-a")]
        self.assertEqual(supportable_scope(records)["comparable_groups"], ["lang-a"])
        stricter = supportable_scope(records, {"min_group_complete": 2})
        self.assertEqual(stricter["not_comparable_groups"], ["lang-a"])


class BuildReviewTests(unittest.TestCase):
    def test_review_is_deterministic_and_carries_basis(self):
        records = [make("a"), make("b", ("outcome_score",), region="east")]
        first = build_review(records, basis={"card_id": "card-1", "minimum_score": 0.6})
        second = build_review(records, basis={"card_id": "card-1", "minimum_score": 0.6})
        self.assertEqual(stable_hash(first), stable_hash(second))
        self.assertEqual(first["basis"]["card_id"], "card-1")
        self.assertEqual(first["schema"], "evidence-review/1")

    def test_limitations_state_observations_and_assumptions(self):
        records = [
            make("a1", language_group="lang-a"),
            make("b1", ("baseline_score", "outcome_score"), language_group="lang-b", region="east"),
        ]
        review = build_review(records)
        notes = "；".join(review["support"]["limitations"])
        self.assertIn("无法比较", notes)
        self.assertIn("未自动采用", notes)
        self.assertIn("published_zero_fill", notes)


if __name__ == "__main__":
    unittest.main()

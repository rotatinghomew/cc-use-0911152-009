from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from learning_evidence.analysis_service import Application, run, verify
from learning_evidence.programs.contract import ContractError, Decision, DomainRecord

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "missing_pattern.json"


def school(rid: str, language: str, consent: float, region: str = "north",
           missing: tuple[str, ...] = (), **overrides) -> DomainRecord:
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
        "language_group": language,
        "consent_rate": consent,
        "region": region,
    }
    for field in missing:
        payload.pop(field, None)
    payload.update(overrides)
    return DomainRecord(rid, payload)


class EvidenceReviewCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.app = Application(self.root / "db.sqlite3")
        self.records = [
            school("a1", "lang-x", 0.9),
            school("a2", "lang-x", 0.9, missing=("baseline_score", "outcome_score")),
            school("b1", "lang-y", 0.9, region="east",
                   missing=("baseline_score", "outcome_score", "usage_minutes")),
            school("c1", "lang-z", 0.3, region="south"),
        ]
        decisions = {}
        for record in self.records:
            self.app.ingest(record)
            decisions[record.record_id] = Decision(record.record_id, "approve", 0.8, ("人工复核",), (), {})
            self.app.repository.save_decision(decisions[record.record_id])
        self.card = self.app.repository.publish_cohort_card(
            "card", "cohort", [record.record_id for record in self.records], decisions)
        self.app.scale_reviews.open("round", "card", 0.6, 5)

    def test_review_gates_approval_by_scope(self):
        attached = self.app.scale_reviews.review_evidence("round")
        self.assertTrue(attached["changed"])
        scope = attached["review"]["scope"]["schools"]
        self.assertEqual(scope["b1"]["reasons"], ["group_not_comparable"])
        self.assertEqual(scope["c1"]["reasons"], ["consent_out_of_scope"])
        with self.assertRaisesRegex(ContractError, "无法比较的群体不能直接判为公平"):
            self.app.scale_reviews.decide("round", "b1", "approved", "评议通过", 1)
        with self.assertRaisesRegex(ContractError, "同意率低于同意范围下限"):
            self.app.scale_reviews.decide("round", "c1", "approved", "评议通过", 1)
        self.app.scale_reviews.decide("round", "b1", "deferred", "证据不足，暂缓")
        self.app.scale_reviews.decide("round", "a1", "approved", "证据及支持准备充分", 1)
        queue = {row["school_id"]: row for row in self.app.scale_reviews.inspect("round")["schools"]}
        self.assertEqual(queue["b1"]["status"], "deferred")
        self.assertEqual(queue["a1"]["status"], "approved")

    def test_review_is_opt_in_and_replay_is_idempotent(self):
        fresh = Application(self.root / "other.sqlite3")
        decisions = {}
        record = school("solo", "lang-solo", 0.9)
        fresh.ingest(record)
        decisions["solo"] = Decision("solo", "approve", 0.9, ("人工复核",), (), {})
        fresh.repository.save_decision(decisions["solo"])
        fresh.repository.publish_cohort_card("card", "cohort", ["solo"], decisions)
        fresh.scale_reviews.open("round", "card", 0.6, 1)
        fresh.scale_reviews.decide("round", "solo", "approved", "未附加审查时按既有口径评议", 1)
        first = self.app.scale_reviews.review_evidence("round")
        second = self.app.scale_reviews.review_evidence("round")
        self.assertTrue(first["changed"])
        self.assertFalse(second["changed"])
        self.assertEqual(first["review_version"], second["review_version"])
        self.assertEqual(self.app.scale_reviews.latest_review("round")["review"], first["review"])

    def test_re_review_separates_estimate_scope_and_eligibility_changes(self):
        self.app.scale_reviews.review_evidence("round")
        self.app.ingest(school("b1", "lang-y", 0.9, region="east"))   # 补报缺失成绩与使用记录
        self.app.ingest(school("c1", "lang-z", 0.9, region="south"))  # 同意更正
        report = self.app.scale_reviews.re_review("round")
        self.assertFalse(report["unchanged"])
        self.assertEqual(report["previous_review_version"], 1)
        self.assertEqual(report["review_version"], 2)
        self.assertEqual(report["cutoff"],
                         {"card_id": "card", "minimum_score": 0.6, "seats": 5, "unchanged": True})
        self.assertEqual([item["school_id"] for item in report["supplementary_reports"]], ["b1"])
        self.assertEqual(report["supplementary_reports"][0]["before_pattern"], "missing_outcome_and_usage")
        self.assertEqual(report["supplementary_reports"][0]["after_pattern"], "complete")
        self.assertTrue(any(item["strategy"] == "complete_case" and item["metric"] == "effectiveness"
                            for item in report["estimate_changes"]))
        self.assertEqual(report["scope_changes"], {"entered": ["b1", "c1"], "exited": []})
        flips = {item["school_id"]: (item["before_approvable"], item["after_approvable"])
                 for item in report["eligibility_changes"]}
        self.assertEqual(flips, {"b1": (False, True), "c1": (False, True)})
        self.assertEqual(report["pending_re-reviewed"], ["b1", "c1"])
        # 已发布评分卡截止口径不变：卡内容、门槛与名额均未改写
        self.assertEqual(self.app.repository.cohort_card("card"), self.card)
        self.assertEqual(self.app.scale_reviews.inspect("round")["minimum_score"], 0.6)
        # 资格变化不自动批准，仍需人工 decide
        self.app.scale_reviews.decide("round", "b1", "approved", "补报后证据充分", 1)
        self.app.scale_reviews.decide("round", "c1", "approved", "同意更正后评议通过", 1)

    def test_re_review_without_new_evidence_reports_no_changes(self):
        self.app.scale_reviews.review_evidence("round")
        report = self.app.scale_reviews.re_review("round")
        self.assertTrue(report["unchanged"])
        self.assertEqual(report["estimate_changes"], [])
        self.assertEqual(report["scope_changes"], {"entered": [], "exited": []})
        self.assertEqual(report["eligibility_changes"], [])

    def test_qualification_correction_is_visible_without_scope_change(self):
        self.app.scale_reviews.review_evidence("round")
        self.app.ingest(school("a1", "lang-x", 0.9, learner_count=12.0))  # 资格相关字段更正
        report = self.app.scale_reviews.re_review("round")
        self.assertFalse(report["unchanged"])
        self.assertEqual(report["supplementary_reports"], [])
        self.assertEqual(report["eligibility_changes"], [])
        notes = {item["school_id"] for item in report["qualification_notes"]}
        self.assertIn("a1", notes)

    def test_completed_conclusions_and_cutoff_survive_re_review(self):
        self.app.scale_reviews.review_evidence("round")
        self.app.scale_reviews.decide("round", "a1", "approved", "证据及支持准备充分", 1)
        self.app.scale_reviews.decide("round", "a2", "deferred", "等候后续名额")
        self.app.ingest(school("b1", "lang-y", 0.9, region="east"))
        report = self.app.scale_reviews.re_review("round")
        self.assertIn("b1", report["scope_changes"]["entered"])
        queue = {row["school_id"]: row for row in self.app.scale_reviews.inspect("round")["schools"]}
        self.assertEqual((queue["a1"]["status"], queue["a1"]["seats"]), ("approved", 1))
        self.assertEqual(queue["a2"]["status"], "deferred")
        with self.assertRaises(ContractError):
            self.app.scale_reviews.decide("round", "a1", "deferred", "改写旧结论")

    def test_insufficient_score_keeps_published_cutoff_after_corrections(self):
        decisions = {
            "low": Decision("low", "approve", 0.3, ("人工复核",), (), {}),
            "high": Decision("high", "approve", 0.9, ("人工复核",), (), {}),
        }
        for rid, consent in (("low", 0.9), ("high", 0.9)):
            record = school(rid, "lang-x", consent)
            self.app.ingest(record)
            self.app.repository.save_decision(decisions[rid])
        self.app.repository.publish_cohort_card("card-2", "cohort", ["low", "high"], decisions)
        self.app.scale_reviews.open("round-2", "card-2", 0.6, 2)
        self.app.scale_reviews.review_evidence("round-2")
        self.app.ingest(school("low", "lang-x", 0.9, outcome_score=99.0))  # 补报提高当前估计
        report = self.app.scale_reviews.re_review("round-2")
        self.assertTrue(report["estimate_changes"])
        queue = {row["school_id"]: row for row in self.app.scale_reviews.inspect("round-2")["schools"]}
        self.assertEqual(queue["low"]["status"], "insufficient")  # 截止口径保留，不因补报改判
        self.assertEqual(queue["low"]["score"], 0.3)
        with self.assertRaises(ContractError):
            self.app.scale_reviews.decide("round-2", "low", "approved", "补报后尝试放行", 1)


class MissingPatternFixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.result = run(FIXTURE, self.root / "run.sqlite3", self.root / "output")

    def test_run_exports_evidence_review_beyond_scorecard(self):
        review = self.result["evidence_review"]
        self.assertEqual(review["basis"]["run_id"], self.result["run_id"])
        counts = review["diagnostics"]["pattern_counts"]
        self.assertEqual(counts["complete"], 3)
        self.assertEqual(counts["missing_outcome"], 2)
        self.assertEqual(counts["missing_usage"], 2)
        self.assertEqual(counts["missing_outcome_and_usage"], 1)
        self.assertEqual(review["scope"]["not_comparable_groups"], ["language_group-3"])
        self.assertNotIn("school-east-1", review["scope"]["supportable_schools"])
        self.assertNotIn("school-south-2", review["scope"]["supportable_schools"])
        self.assertIsNone(review["sensitivity"]["selection"])
        bundle = self.root / "output"
        self.assertTrue((bundle / "evidence-review.json").is_file())
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("evidence-review.json", manifest["files"])
        self.assertTrue(verify(self.root / "run.sqlite3", bundle)["ok"])

    def test_campaign_on_run_card_uses_review_scope(self):
        app = Application(self.root / "run.sqlite3")
        card_id = self.result["cohort_card"]["card_id"]
        app.scale_reviews.open("round", card_id, 0.5, 4)
        attached = app.scale_reviews.review_evidence("round")
        scope = attached["review"]["scope"]
        self.assertEqual(attached["review"]["basis"]["card_id"], card_id)
        flags = {row["school_id"]: row["review_flags"]
                 for row in app.scale_reviews.inspect("round")["schools"]}
        self.assertFalse(flags["school-east-1"]["approvable"])
        self.assertFalse(flags["school-south-2"]["approvable"])
        self.assertTrue(flags["school-north-1"]["approvable"])
        with self.assertRaises(ContractError):
            app.scale_reviews.decide("round", "school-east-1", "approved", "评议通过", 1)


if __name__ == "__main__":
    unittest.main()

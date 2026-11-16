"""使用已发布评分卡安排学校扩容评议及名额确认。

扩容证据审查（review_evidence / re_review）在评分卡之外附加缺失模式诊断与
敏感性比较：判断可支持的学校范围，无法比较的群体不能直接判为公平。
资格或同意更正后通过 re_review 重审待批申请；已发布评分卡的截止口径保持不变，
补报带来的估计变化、适用范围变化与扩容资格变化分别报告。
"""
import json
import math

from learning_evidence.programs.contract import (
    ContractError,
    DomainRecord,
    canonical_json,
    stable_hash,
    utc_now,
)
from learning_evidence.analytics.missingness import (
    SCOPE_REASON_MESSAGES,
    STRATEGY_ORDER,
    build_review,
)

_ESTIMATE_METRICS = ("effectiveness", "fairness", "equity_gap")


class ScaleReviews:
    def __init__(self, repository):
        self.repository = repository
        with repository.connect() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS scale_campaigns(
                    campaign_id TEXT PRIMARY KEY, card_id TEXT NOT NULL,
                    evidence_hash TEXT NOT NULL, minimum_score REAL NOT NULL,
                    seats INTEGER NOT NULL CHECK(seats>=0));
                CREATE TABLE IF NOT EXISTS school_scale_reviews(
                    campaign_id TEXT NOT NULL, school_id TEXT NOT NULL,
                    score REAL NOT NULL, source_outcome TEXT NOT NULL, status TEXT NOT NULL,
                    seats INTEGER NOT NULL DEFAULT 0, reason TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(campaign_id,school_id));
                CREATE TABLE IF NOT EXISTS scale_evidence_reviews(
                    campaign_id TEXT NOT NULL, review_version INTEGER NOT NULL,
                    review_hash TEXT NOT NULL, review_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(campaign_id,review_version));
                CREATE TABLE IF NOT EXISTS scale_review_flags(
                    campaign_id TEXT NOT NULL, school_id TEXT NOT NULL,
                    review_version INTEGER NOT NULL, approvable INTEGER NOT NULL,
                    reasons_json TEXT NOT NULL,
                    PRIMARY KEY(campaign_id,school_id,review_version));
            """)

    def open(self, campaign_id, card_id, minimum_score, seats):
        if not campaign_id or type(seats) is not int or seats < 0:
            raise ContractError("扩容评议及名额无效")
        if isinstance(minimum_score, bool) or not isinstance(minimum_score, (int, float)) or not math.isfinite(minimum_score) or not 0 <= minimum_score <= 1:
            raise ContractError("候选评分门槛无效")
        with self.repository.transaction() as c:
            from learning_evidence.publication.cohort_scorecard import read_card
            card = read_card(c, card_id)
            old = c.execute("SELECT * FROM scale_campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
            if old:
                if (old["card_id"], old["minimum_score"], old["seats"]) != (card_id, minimum_score, seats):
                    raise ContractError("同一扩容评议不可混用评分卡或门槛")
                return self._read(c, campaign_id)
            c.execute("INSERT INTO scale_campaigns VALUES(?,?,?,?,?)",
                      (campaign_id, card_id, card["evidence_hash"], minimum_score, seats))
            for school in card["schools"]:
                status = "pending" if school["score"] >= minimum_score else "insufficient"
                c.execute("INSERT INTO school_scale_reviews(campaign_id,school_id,score,source_outcome,status) VALUES(?,?,?,?,?)",
                          (campaign_id, school["school_id"], school["score"], school["outcome"], status))
            return self._read(c, campaign_id)

    def decide(self, campaign_id, school_id, status, reason, seats=0):
        if status not in {"approved", "deferred"} or not isinstance(reason, str) or not reason.strip():
            raise ContractError("评议需要明确结论及依据")
        if type(seats) is not int or seats < 0 or (status == "deferred" and seats != 0):
            raise ContractError("名额无效")
        with self.repository.transaction() as c:
            campaign = c.execute("SELECT * FROM scale_campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
            row = c.execute("SELECT * FROM school_scale_reviews WHERE campaign_id=? AND school_id=?",
                            (campaign_id, school_id)).fetchone()
            if campaign is None or row is None:
                raise ContractError("学校不属于本次比较人口")
            if row["status"] != "pending":
                if (row["status"], row["reason"], row["seats"]) == (status, reason, seats):
                    return
                raise ContractError("评分不足或评议已完成，不可覆盖历史")
            if status == "approved":
                if row["source_outcome"] in {"hold", "reject"} or seats == 0:
                    raise ContractError("既有安全门禁未放行或未申请扩容名额")
                flag = self._latest_flag(c, campaign_id, school_id)
                if flag is not None and not flag["approvable"]:
                    reasons = "；".join(SCOPE_REASON_MESSAGES.get(item, item)
                                        for item in flag["reasons"]) or "证据审查未支持"
                    raise ContractError(f"证据审查未支持批准：{reasons}")
                allocated = c.execute("SELECT COALESCE(SUM(seats),0) FROM school_scale_reviews WHERE campaign_id=?",
                                      (campaign_id,)).fetchone()[0]
                if allocated + seats > campaign["seats"]:
                    raise ContractError("本轮扩容名额不足")
            c.execute("UPDATE school_scale_reviews SET status=?,seats=?,reason=? WHERE campaign_id=? AND school_id=?",
                      (status, seats, reason, campaign_id, school_id))

    def review_evidence(self, campaign_id, context=None):
        """为评议附加一份证据审查；内容不变时重放幂等，不新增版本。"""
        with self.repository.transaction() as c:
            campaign = c.execute("SELECT * FROM scale_campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
            if campaign is None:
                raise ContractError("扩容评议不存在")
            from learning_evidence.publication.cohort_scorecard import read_card
            card = read_card(c, campaign["card_id"])
            records = self._load_records(c, [school["school_id"] for school in card["schools"]])
            review = build_review(records, dict(context or {}),
                                  basis={"card_id": card["card_id"],
                                         "card_evidence_hash": card["evidence_hash"],
                                         "minimum_score": campaign["minimum_score"],
                                         "seats": campaign["seats"]},
                                  expected_schools=[school["school_id"] for school in card["schools"]])
            review_hash = stable_hash(review)
            latest = c.execute("SELECT review_version,review_hash,review_json FROM scale_evidence_reviews "
                               "WHERE campaign_id=? ORDER BY review_version DESC LIMIT 1",
                               (campaign_id,)).fetchone()
            if latest and latest["review_hash"] == review_hash:
                return {"campaign_id": campaign_id, "review_version": latest["review_version"],
                        "changed": False, "review": json.loads(latest["review_json"])}
            version = (latest["review_version"] if latest else 0) + 1
            c.execute("INSERT INTO scale_evidence_reviews VALUES(?,?,?,?,?)",
                      (campaign_id, version, review_hash, canonical_json(review), utc_now()))
            for school_id, row in review["scope"]["schools"].items():
                c.execute("INSERT INTO scale_review_flags VALUES(?,?,?,?,?)",
                          (campaign_id, school_id, version,
                           1 if row["scope_status"] == "supportable" else 0,
                           canonical_json(row["reasons"])))
            return {"campaign_id": campaign_id, "review_version": version,
                    "changed": True, "review": review}

    def re_review(self, campaign_id, context=None):
        """资格或同意更正后重审待批申请。

        已发布评分卡的截止口径（评分、门槛、名额）保持不变；报告把补报带来的
        估计变化、适用范围变化与扩容资格变化分开列出，待批学校仍需 decide 确认。
        """
        with self.repository.connect() as c:
            previous = c.execute("SELECT review_version,review_json FROM scale_evidence_reviews "
                                 "WHERE campaign_id=? ORDER BY review_version DESC LIMIT 1",
                                 (campaign_id,)).fetchone()
        attached = self.review_evidence(campaign_id, context)
        with self.repository.connect() as c:
            campaign = c.execute("SELECT * FROM scale_campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
            queue = {row["school_id"]: row for row in c.execute(
                "SELECT * FROM school_scale_reviews WHERE campaign_id=?", (campaign_id,))}
        cutoff = {"card_id": campaign["card_id"], "minimum_score": campaign["minimum_score"],
                  "seats": campaign["seats"], "unchanged": True}
        if previous is None:
            return {"campaign_id": campaign_id, "initial": True, "unchanged": False,
                    "review_version": attached["review_version"], "previous_review_version": None,
                    "cutoff": cutoff, "supplementary_reports": [], "estimate_changes": [],
                    "scope_changes": {"entered": [], "exited": []}, "eligibility_changes": [],
                    "qualification_notes": [], "pending_re-reviewed": []}
        if not attached["changed"]:
            return {"campaign_id": campaign_id, "initial": False, "unchanged": True,
                    "review_version": attached["review_version"],
                    "previous_review_version": previous["review_version"],
                    "cutoff": cutoff, "supplementary_reports": [], "estimate_changes": [],
                    "scope_changes": {"entered": [], "exited": []}, "eligibility_changes": [],
                    "qualification_notes": [], "pending_re-reviewed": []}
        before = json.loads(previous["review_json"])
        after = attached["review"]
        report = {"campaign_id": campaign_id, "initial": False, "unchanged": False,
                  "review_version": attached["review_version"],
                  "previous_review_version": previous["review_version"], "cutoff": cutoff}
        report["supplementary_reports"] = self._pattern_changes(before, after)
        report["estimate_changes"] = self._estimate_changes(before, after)
        report["scope_changes"] = self._scope_changes(before, after)
        report["eligibility_changes"] = self._eligibility_changes(before, after, queue)
        report["qualification_notes"] = self._qualification_notes(before, after)
        changed_schools = {item["school_id"] for item in report["supplementary_reports"]}
        changed_schools |= {item["school_id"] for item in report["eligibility_changes"]}
        changed_schools |= set(report["scope_changes"]["entered"]) | set(report["scope_changes"]["exited"])
        report["pending_re-reviewed"] = sorted(
            school for school in changed_schools
            if queue.get(school) is not None and queue[school]["status"] == "pending")
        return report

    def latest_review(self, campaign_id):
        with self.repository.connect() as c:
            row = c.execute("SELECT review_version,review_json FROM scale_evidence_reviews "
                            "WHERE campaign_id=? ORDER BY review_version DESC LIMIT 1",
                            (campaign_id,)).fetchone()
        if row is None:
            return None
        return {"campaign_id": campaign_id, "review_version": row["review_version"],
                "review": json.loads(row["review_json"])}

    @staticmethod
    def _load_records(c, school_ids):
        records = []
        for school_id in sorted(set(school_ids)):
            row = c.execute("SELECT * FROM records WHERE record_id=?", (school_id,)).fetchone()
            if row is not None:
                records.append(DomainRecord(row["record_id"], json.loads(row["payload_json"]),
                                            row["state"], row["version"], row["occurred_at"],
                                            row["source"]))
        return records

    @staticmethod
    def _latest_flag(c, campaign_id, school_id):
        row = c.execute("SELECT approvable,reasons_json FROM scale_review_flags "
                        "WHERE campaign_id=? AND school_id=? ORDER BY review_version DESC LIMIT 1",
                        (campaign_id, school_id)).fetchone()
        if row is None:
            return None
        return {"approvable": bool(row["approvable"]), "reasons": json.loads(row["reasons_json"])}

    @staticmethod
    def _pattern_changes(before, after):
        def patterns(review):
            return {row["school_id"]: row["pattern"] for row in review["diagnostics"]["schools"]}
        old, new = patterns(before), patterns(after)
        return [{"school_id": school, "before_pattern": old.get(school), "after_pattern": new[school]}
                for school in sorted(new) if old.get(school) != new[school]]

    @staticmethod
    def _estimate_changes(before, after):
        changes = []
        for strategy in STRATEGY_ORDER:
            old = before["sensitivity"]["strategies"].get(strategy, {})
            new = after["sensitivity"]["strategies"].get(strategy, {})
            for metric in _ESTIMATE_METRICS:
                previous, current = old.get(metric), new.get(metric)
                if previous != current:
                    changes.append({"strategy": strategy, "metric": metric,
                                    "before": previous, "after": current,
                                    "delta": None if previous is None or current is None
                                    else current - previous})
        return changes

    @staticmethod
    def _scope_changes(before, after):
        old = set(before["scope"]["supportable_schools"])
        new = set(after["scope"]["supportable_schools"])
        return {"entered": sorted(new - old), "exited": sorted(old - new)}

    @staticmethod
    def _eligibility_changes(before, after, queue):
        old = before["scope"]["schools"]
        new = after["scope"]["schools"]
        changes = []
        for school in sorted(set(old) | set(new)):
            previous = old.get(school, {})
            current = new.get(school, {})
            if (previous.get("scope_status"), previous.get("reasons")) != \
               (current.get("scope_status"), current.get("reasons")):
                row = queue.get(school)
                changes.append({
                    "school_id": school,
                    "queue_status": row["status"] if row else None,
                    "before_approvable": previous.get("scope_status") == "supportable",
                    "after_approvable": current.get("scope_status") == "supportable",
                    "before_reasons": previous.get("reasons", ["record_missing"]),
                    "after_reasons": current.get("reasons", ["record_missing"])})
        return changes

    @staticmethod
    def _qualification_notes(before, after):
        def eligibility(review):
            return {row["school_id"]: row["eligibility"] for row in review["diagnostics"]["schools"]}
        old, new = eligibility(before), eligibility(after)
        notes = []
        for school in sorted(set(old) | set(new)):
            previous, current = old.get(school), new.get(school)
            if previous != current:
                notes.append({"school_id": school, "before": previous, "after": current})
        return notes

    def _read(self, c, campaign_id):
        campaign = c.execute("SELECT * FROM scale_campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
        if campaign is None:
            raise ContractError("扩容评议不存在")
        version_row = c.execute("SELECT MAX(review_version) FROM scale_evidence_reviews WHERE campaign_id=?",
                                (campaign_id,)).fetchone()
        schools = []
        for row in c.execute("SELECT * FROM school_scale_reviews WHERE campaign_id=? ORDER BY score DESC,school_id",
                             (campaign_id,)):
            schools.append(dict(row) | {"review_flags": self._latest_flag(c, campaign_id, row["school_id"])})
        return dict(campaign) | {"schools": schools,
                                 "latest_review_version": version_row[0] if version_row else None}

    def inspect(self, campaign_id):
        with self.repository.connect() as c:
            return self._read(c, campaign_id)

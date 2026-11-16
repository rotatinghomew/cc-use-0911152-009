"""缺失模式诊断、敏感性比较与可支持范围判断。

本模块在评分卡之外回答三个问题：

1. 哪些学校缺少成绩（baseline_score/outcome_score）、哪些缺少使用记录
   （usage_minutes），缺失是否与资格、同意范围、地区差异相关；
2. 成效与公平性估计在多种缺失处理假设下如何变化（敏感性比较），
   不自动采用最有利的处理方法；
3. 哪些学校属于可支持范围：所在群体可比较、同意范围充分、证据在库。
   无法比较的群体不能直接判为公平。

所有函数都是纯函数，结果不含时间戳，可安全地做稳定哈希与重放核对。
"""
from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping, Sequence

from learning_evidence.programs.contract import (
    REQUIRED_FIELDS,
    ContractError,
    DomainRecord,
    bounded,
    finite_number,
    normalized_text,
    require_fields,
)
from learning_evidence.analytics.bias_adjustment import adjusted_effect, equity_summary
from learning_evidence.quality.eligibility import evaluate, summarize

# 成绩与使用记录是项目负责人点名的两类缺失证据。
OUTCOME_FIELDS = ("baseline_score", "outcome_score")
USAGE_FIELDS = ("usage_minutes",)

# 进入成效/公平性估计的数值证据字段，敏感性处理只填充这些字段。
NUMERIC_EVIDENCE_FIELDS = (
    "learner_count",
    "usage_minutes",
    "baseline_score",
    "outcome_score",
    "missing_rate",
    "teacher_support",
    "connectivity",
    "consent_rate",
)

# 估计“越有利”时各字段应取的方向：True 表示取观测最大值更有利。
FAVORABLE_HIGH = {
    "learner_count": True,
    "usage_minutes": True,
    "baseline_score": False,
    "outcome_score": True,
    "missing_rate": False,
    "teacher_support": False,
    "connectivity": True,
    "consent_rate": True,
}

STRATEGY_ORDER = (
    "complete_case",
    "published_zero_fill",
    "consent_restricted",
    "stratum_mean",
    "best_case",
    "worst_case",
)

STRATEGY_ASSUMPTIONS = {
    "complete_case": "只使用所有必填字段完整的学校；隐含假设缺失与估计目标无关",
    "published_zero_fill": "已发布评分卡口径：缺失数值按 0 处理，隐含假设缺失证据无贡献",
    "consent_restricted": "仅保留同意范围内的学校，其余按已发布口径处理",
    "stratum_mean": "缺失数值用同地区（或代理分组）观测均值填补",
    "best_case": "缺失数值按对估计最有利的观测极值填补（上界）",
    "worst_case": "缺失数值按对估计最不利的观测极值填补（下界）",
}

SCOPE_REASON_MESSAGES = {
    "record_missing": "学校记录不在当前证据库",
    "group_not_comparable": "所在群体缺少完整观测，无法比较的群体不能直接判为公平",
    "consent_unknown": "同意率缺失，同意范围未知",
    "consent_out_of_scope": "同意率低于同意范围下限",
}

_CONSENT_UNKNOWN = "unknown"
_CONSENT_LIMITED = "limited"
_CONSENT_PARTIAL = "partial"
_CONSENT_FULL = "full"


def _context_float(context: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return finite_number(context.get(key, default))
    except (ContractError, ValueError, TypeError):
        return default


def _is_missing(record: DomainRecord, field: str) -> bool:
    return field not in record.payload or record.payload[field] in (None, "")


def missing_pattern(record: DomainRecord) -> dict[str, Any]:
    """单所学校的缺失模式：缺成绩、缺使用记录或两者皆缺。"""
    missing = set(require_fields(record))
    outcome = sorted(field for field in OUTCOME_FIELDS if field in missing)
    usage = sorted(field for field in USAGE_FIELDS if field in missing)
    other = sorted(missing - set(OUTCOME_FIELDS) - set(USAGE_FIELDS))
    if not missing:
        label = "complete"
    elif outcome and usage:
        label = "missing_outcome_and_usage"
    elif outcome:
        label = "missing_outcome"
    elif usage:
        label = "missing_usage"
    else:
        label = "missing_other"
    return {
        "school_id": record.record_id,
        "pattern": label,
        "missing_fields": sorted(missing),
        "missing_outcome_fields": outcome,
        "missing_usage_fields": usage,
        "missing_other_fields": other,
    }


def consent_status(record: DomainRecord, context: Mapping[str, Any]) -> str:
    """同意范围分档：unknown / limited / partial / full。"""
    if _is_missing(record, "consent_rate"):
        return _CONSENT_UNKNOWN
    value = finite_number(record.payload.get("consent_rate"))
    floor = _context_float(context, "consent_floor", 0.5)
    full = _context_float(context, "consent_full", 0.8)
    if value < floor:
        return _CONSENT_LIMITED
    if value < full:
        return _CONSENT_PARTIAL
    return _CONSENT_FULL


def consent_in_scope(record: DomainRecord, context: Mapping[str, Any]) -> bool:
    return consent_status(record, context) in {_CONSENT_PARTIAL, _CONSENT_FULL}


def rule_eligibility(record: DomainRecord, context: Mapping[str, Any]) -> dict[str, Any]:
    """资格分层：规则目录的 critical/failed 计数及按上限判定的资格状态。"""
    summary = summarize(evaluate(record, context))
    ceiling = int(_context_float(context, "eligibility_critical_ceiling", 0.0))
    return {
        "critical": summary["critical"],
        "failed": summary["failed"],
        "status": "eligible" if summary["critical"] <= ceiling else "ineligible",
    }


def _region_field(records: Sequence[DomainRecord], context: Mapping[str, Any]) -> tuple[str, bool]:
    configured = str(context.get("region_field") or "region")
    if any(not _is_missing(record, configured) for record in records):
        return configured, configured != "region"
    return "language_group", True


def _stratum(record: DomainRecord, field: str) -> str:
    if _is_missing(record, field):
        return "unknown"
    return normalized_text(record.payload.get(field)) or "unknown"


def diagnose(records: Sequence[DomainRecord], context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """缺失模式诊断：按资格、同意范围、地区分层，检验缺失是否与这些维度相关。"""
    context = dict(context or {})
    records = list(records)
    patterns = {record.record_id: missing_pattern(record) for record in records}
    region_field, region_proxy = _region_field(records, context)
    tolerance = _context_float(context, "missingness_association_tolerance", 0.25)

    schools = []
    for record in records:
        pattern = patterns[record.record_id]
        schools.append({
            "school_id": record.record_id,
            "pattern": pattern["pattern"],
            "missing_fields": pattern["missing_fields"],
            "consent_status": consent_status(record, context),
            "eligibility": rule_eligibility(record, context),
            "region": _stratum(record, region_field),
            "language_group": _stratum(record, "language_group"),
        })

    pattern_counts: dict[str, int] = {}
    field_missing: dict[str, int] = {field: 0 for field in REQUIRED_FIELDS}
    group_missing = {"outcome": [], "usage": []}
    for row in schools:
        pattern_counts[row["pattern"]] = pattern_counts.get(row["pattern"], 0) + 1
        for field in row["missing_fields"]:
            field_missing[field] += 1
        pattern = patterns[row["school_id"]]
        if pattern["missing_outcome_fields"]:
            group_missing["outcome"].append(row["school_id"])
        if pattern["missing_usage_fields"]:
            group_missing["usage"].append(row["school_id"])

    dimensions: dict[str, Any] = {}
    dimension_keys = {
        "eligibility": lambda row: row["eligibility"]["status"],
        "consent_scope": lambda row: row["consent_status"],
        "region": lambda row: row["region"],
    }
    for name, key_fn in dimension_keys.items():
        strata: dict[str, dict[str, Any]] = {}
        for row in schools:
            label = key_fn(row)
            bucket = strata.setdefault(label, {"records": 0, "missing_any": 0,
                                               "missing_outcome": 0, "missing_usage": 0})
            bucket["records"] += 1
            pattern = patterns[row["school_id"]]
            if pattern["missing_fields"]:
                bucket["missing_any"] += 1
            if pattern["missing_outcome_fields"]:
                bucket["missing_outcome"] += 1
            if pattern["missing_usage_fields"]:
                bucket["missing_usage"] += 1
        rates = []
        for label, bucket in sorted(strata.items()):
            count = max(1, bucket["records"])
            bucket["missing_rate"] = bucket["missing_any"] / count
            bucket["outcome_missing_rate"] = bucket["missing_outcome"] / count
            bucket["usage_missing_rate"] = bucket["missing_usage"] / count
            rates.append(bucket["missing_rate"])
        gap = max(rates) - min(rates) if rates else 0.0
        dimensions[name] = {"strata": strata, "max_missing_rate_gap": gap,
                            "associated": bool(gap >= tolerance and len(rates) > 1)}

    return {
        "record_count": len(records),
        "pattern_counts": pattern_counts,
        "field_missing_counts": field_missing,
        "missing_outcome_schools": sorted(group_missing["outcome"]),
        "missing_usage_schools": sorted(group_missing["usage"]),
        "region_field": region_field,
        "region_is_proxy": region_proxy,
        "dimensions": dimensions,
        "schools": schools,
    }


def _fill(record: DomainRecord, fills: Mapping[str, float]) -> DomainRecord:
    payload = dict(record.payload)
    for field, value in fills.items():
        if field not in payload or payload[field] in (None, ""):
            payload[field] = value
    return DomainRecord(record.record_id, payload, record.state, record.version,
                        record.occurred_at, record.source)


def _observed_values(records: Sequence[DomainRecord], field: str) -> list[float]:
    values = []
    for record in records:
        if not _is_missing(record, field):
            values.append(finite_number(record.payload.get(field)))
    return values


def _strategy_rows(records: Sequence[DomainRecord], strategy: str,
                   context: Mapping[str, Any], region_field: str) -> list[DomainRecord]:
    records = list(records)
    if strategy == "complete_case":
        return [record for record in records if not require_fields(record)]
    if strategy == "published_zero_fill":
        return [_fill(record, {field: 0.0 for field in NUMERIC_EVIDENCE_FIELDS}) for record in records]
    if strategy == "consent_restricted":
        return [_fill(record, {field: 0.0 for field in NUMERIC_EVIDENCE_FIELDS})
                for record in records if consent_in_scope(record, context)]
    if strategy == "stratum_mean":
        overall = {field: (statistics.fmean(_observed_values(records, field))
                           if _observed_values(records, field) else 0.0)
                   for field in NUMERIC_EVIDENCE_FIELDS}
        by_region: dict[str, dict[str, float]] = {}
        for field in NUMERIC_EVIDENCE_FIELDS:
            grouped: dict[str, list[float]] = {}
            for record in records:
                if not _is_missing(record, field):
                    grouped.setdefault(_stratum(record, region_field), []).append(
                        finite_number(record.payload.get(field)))
            for label, values in grouped.items():
                by_region.setdefault(label, {})[field] = statistics.fmean(values)
        filled = []
        for record in records:
            label = _stratum(record, region_field)
            means = by_region.get(label, {})
            fills = {field: means.get(field, overall[field]) for field in NUMERIC_EVIDENCE_FIELDS}
            filled.append(_fill(record, fills))
        return filled
    if strategy in {"best_case", "worst_case"}:
        fills_by_field: dict[str, float] = {}
        for field in NUMERIC_EVIDENCE_FIELDS:
            observed = _observed_values(records, field)
            if not observed:
                fills_by_field[field] = 0.0
                continue
            high = FAVORABLE_HIGH[field] if strategy == "best_case" else not FAVORABLE_HIGH[field]
            fills_by_field[field] = max(observed) if high else min(observed)
        return [_fill(record, fills_by_field) for record in records]
    raise ContractError(f"unknown sensitivity strategy: {strategy}")


def _estimands(rows: Sequence[DomainRecord]) -> dict[str, Any]:
    if not rows:
        return {"defined": False, "effectiveness": None, "fairness": None, "equity_gap": None}
    effects = [adjusted_effect(row)["adjusted"] for row in rows]
    equity = equity_summary(rows)
    return {"defined": True,
            "effectiveness": statistics.fmean(effects),
            "fairness": equity["fairness"],
            "equity_gap": equity["gap"]}


def sensitivity(records: Sequence[DomainRecord], context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """敏感性比较：同一批学校在多种缺失处理假设下的成效与公平性估计。

    结果并列展示，不自动采用最有利的处理方法；已发布评分卡截止口径对应
    published_zero_fill。
    """
    context = dict(context or {})
    records = list(records)
    region_field, _ = _region_field(records, context)
    strategies: dict[str, Any] = {}
    for name in STRATEGY_ORDER:
        rows = _strategy_rows(records, name, context, region_field)
        estimands = _estimands(rows)
        strategies[name] = {
            "assumption": STRATEGY_ASSUMPTIONS[name],
            "included_count": len(rows),
            "included_schools": sorted(row.record_id for row in rows),
            **estimands,
        }

    def _range(metric: str) -> dict[str, Any]:
        values = [row[metric] for row in strategies.values() if row["defined"]]
        if not values:
            return {"min": None, "max": None}
        return {"min": min(values), "max": max(values)}

    published = strategies["published_zero_fill"]
    effectiveness_range = _range("effectiveness")
    fairness_range = _range("fairness")

    def _within(value: float | None, bounds: dict[str, Any]) -> bool | None:
        if value is None or bounds["min"] is None:
            return None
        return bool(bounds["min"] <= value <= bounds["max"])

    return {
        "strategies": strategies,
        "effectiveness_range": effectiveness_range,
        "fairness_range": fairness_range,
        "published_basis": "published_zero_fill",
        "published_within_range": {
            "effectiveness": _within(published["effectiveness"], effectiveness_range),
            "fairness": _within(published["fairness"], fairness_range),
        },
        "selection": None,
        "selection_note": "各处理假设并列展示，未自动采用最有利的处理方法",
    }


def supportable_scope(records: Sequence[DomainRecord], context: Mapping[str, Any] | None = None,
                      expected_schools: Iterable[str] | None = None) -> dict[str, Any]:
    """可支持范围：群体可比较 + 同意范围充分 + 证据在库。

    无法比较的群体（组内没有任何完整观测）不能直接判为公平。
    """
    context = dict(context or {})
    records = list(records)
    group_field = str(context.get("group_field") or "language_group")
    minimum = max(1, int(_context_float(context, "min_group_complete", 1.0)))

    complete_by_group: dict[str, int] = {}
    for record in records:
        if not require_fields(record):
            group = _stratum(record, group_field)
            complete_by_group[group] = complete_by_group.get(group, 0) + 1

    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        group = _stratum(record, group_field)
        reasons: list[str] = []
        if complete_by_group.get(group, 0) < minimum:
            reasons.append("group_not_comparable")
        status = consent_status(record, context)
        if status == _CONSENT_UNKNOWN:
            reasons.append("consent_unknown")
        elif status == _CONSENT_LIMITED:
            reasons.append("consent_out_of_scope")
        rows[record.record_id] = {
            "school_id": record.record_id,
            "group": group,
            "group_complete_count": complete_by_group.get(group, 0),
            "consent_status": status,
            "scope_status": "supportable" if not reasons else "not_supportable",
            "reasons": reasons,
        }

    missing_records: list[str] = []
    for school_id in sorted(set(expected_schools or ())):
        if school_id not in rows:
            rows[school_id] = {
                "school_id": school_id,
                "group": None,
                "group_complete_count": 0,
                "consent_status": None,
                "scope_status": "not_supportable",
                "reasons": ["record_missing"],
            }
            missing_records.append(school_id)

    comparable = sorted(group for group, count in complete_by_group.items() if count >= minimum)
    all_groups = sorted({_stratum(record, group_field) for record in records})
    not_comparable = [group for group in all_groups if group not in comparable]
    supportable = sorted(school for school, row in rows.items() if row["scope_status"] == "supportable")
    return {
        "group_field": group_field,
        "min_group_complete": minimum,
        "comparable_groups": comparable,
        "not_comparable_groups": not_comparable,
        "supportable_schools": supportable,
        "records_missing_from_warehouse": missing_records,
        "schools": {school: rows[school] for school in sorted(rows)},
    }


def conclusion_support(result: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """结论支持度：哪些结论由观测直接支持，哪些依赖处理假设。"""
    context = dict(context or {})
    strategies: Mapping[str, Any] = result.get("strategies", {})
    effect_threshold = _context_float(context, "effect_threshold", 0.0)
    fairness_threshold = _context_float(context, "fairness_threshold", 0.8)

    def _status(metric: str, threshold: float) -> dict[str, Any]:
        holds, fails = [], []
        for name in STRATEGY_ORDER:
            row = strategies.get(name)
            if not row or not row["defined"]:
                continue
            (holds if row[metric] is not None and row[metric] >= threshold else fails).append(name)
        if holds and not fails:
            status = "observation_supported"
        elif holds:
            status = "assumption_dependent"
        else:
            status = "unsupported"
        return {"status": status, "threshold": threshold,
                "holds_under": holds, "fails_under": fails}

    complete_case = strategies.get("complete_case", {})
    observations = complete_case.get("included_schools", [])
    conclusions = {
        "positive_effect": _status("effectiveness", effect_threshold),
        "fair_across_groups": _status("fairness", fairness_threshold),
    }
    for conclusion in conclusions.values():
        conclusion["observations"] = observations
        conclusion["depends_on_assumptions"] = conclusion["status"] == "assumption_dependent"
    return {"conclusions": conclusions}


def _limitations(diagnostics: Mapping[str, Any], scope: Mapping[str, Any],
                 records: Sequence[DomainRecord]) -> list[str]:
    notes: list[str] = []
    if diagnostics.get("region_is_proxy"):
        notes.append("未提供地区字段，地区差异以 language_group 近似")
    labels = {"eligibility": "资格", "consent_scope": "同意范围", "region": "地区"}
    for name, dimension in diagnostics.get("dimensions", {}).items():
        if dimension.get("associated"):
            notes.append(
                f"缺失与{labels.get(name, name)}相关：组间缺失率最大差 "
                f"{dimension['max_missing_rate_gap']:.3f}，完整观测结论可能不代表全部学校")
    for group in scope.get("not_comparable_groups", []):
        notes.append(f"群体 {group} 没有完整观测，无法比较，公平性判断不覆盖该群体")
    if scope.get("records_missing_from_warehouse"):
        notes.append("部分评分卡学校不在当前证据库，相关结论不覆盖这些学校")
    if any(_is_missing(record, "language_group") for record in records):
        notes.append("部分记录缺少 language_group，公平性汇总包含未知组")
    notes.append("敏感性比较并列展示各处理假设，未自动采用最有利的结果；"
                 "已发布评分卡截止口径对应 published_zero_fill")
    return notes


def build_review(records: Sequence[DomainRecord], context: Mapping[str, Any] | None = None,
                 basis: Mapping[str, Any] | None = None,
                 expected_schools: Iterable[str] | None = None) -> dict[str, Any]:
    """汇总缺失诊断、敏感性比较与可支持范围为一份证据审查。

    basis 记录审查所依赖的截止口径（如已发布评分卡的 card_id、门槛），
    便于把补报带来的估计变化与扩容资格变化分清。
    """
    context = dict(context or {})
    records = list(records)
    diagnostics = diagnose(records, context)
    comparison = sensitivity(records, context)
    scope = supportable_scope(records, context, expected_schools)
    support = conclusion_support(comparison, context)
    return {
        "schema": "evidence-review/1",
        "basis": dict(basis or {}),
        "population": {"record_count": len(records),
                       "school_ids": sorted(record.record_id for record in records)},
        "diagnostics": diagnostics,
        "sensitivity": comparison,
        "scope": scope,
        "support": {"conclusions": support["conclusions"],
                    "limitations": _limitations(diagnostics, scope, records)},
    }

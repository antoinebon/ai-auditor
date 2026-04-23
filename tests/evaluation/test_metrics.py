"""Tests for cross-strategy comparison metrics."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_auditor.evaluation.metrics import (
    aggregate,
    cohens_kappa,
    compare_docs,
    jaccard,
    pair_by_control,
)
from ai_auditor.models import (
    ControlAssessment,
    EvidenceSpan,
    Report,
    ReportStats,
)


def _assessment(
    cid: str,
    coverage: str,
    evidence_ids: list[str] | None = None,
) -> ControlAssessment:
    return ControlAssessment(
        control_id=cid,
        coverage=coverage,  # type: ignore[arg-type]
        evidence=[EvidenceSpan(section_id=s, relevance_note="n") for s in (evidence_ids or [])],
        reasoning="...",
        confidence="medium",
    )


def _report(assessments: list[ControlAssessment], doc_name: str = "doc.pdf") -> Report:
    stats = ReportStats(total_controls=len(assessments), covered=0, partial=0, not_covered=0)
    return Report(
        document_path=Path(doc_name),
        document_title=doc_name,
        analyzed_at=datetime(2026, 4, 22, tzinfo=UTC),
        model="test",
        agentic=False,
        assessments=assessments,
        summary="-",
        stats=stats,
    )


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------


def test_jaccard_identical_sets_is_one() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_is_zero() -> None:
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty_is_one() -> None:
    # Convention: empty vs empty counts as agreement. Matters when two
    # assessments both rightly return no evidence (not_covered).
    assert jaccard(set(), set()) == 1.0


def test_jaccard_partial_overlap() -> None:
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(2 / 4)


# ---------------------------------------------------------------------------
# cohens_kappa
# ---------------------------------------------------------------------------


def test_kappa_perfect_agreement_is_one() -> None:
    pairs = [("covered", "covered"), ("partial", "partial"), ("not_covered", "not_covered")]
    assert cohens_kappa(pairs) == pytest.approx(1.0)  # type: ignore[arg-type]


def test_kappa_perfect_disagreement_is_negative() -> None:
    pairs = [
        ("covered", "not_covered"),
        ("partial", "covered"),
        ("not_covered", "partial"),
    ]
    # Marginal distribution is uniform, so p_e = 1/3; observed agreement 0.
    # kappa = (0 - 1/3) / (1 - 1/3) = -0.5
    assert cohens_kappa(pairs) == pytest.approx(-0.5)  # type: ignore[arg-type]


def test_kappa_empty_input_returns_zero() -> None:
    assert cohens_kappa([]) == 0.0


def test_kappa_single_label_all_agree_returns_one() -> None:
    pairs = [("covered", "covered")] * 5
    assert cohens_kappa(pairs) == pytest.approx(1.0)  # type: ignore[arg-type]


def test_kappa_matches_textbook_example() -> None:
    # Classic 2x2-ish scenario adapted to 3 labels: worked out by hand.
    # 10 pairs: 6 agree on 'covered', 2 agree on 'partial', 1 disagree
    # (covered/partial), 1 disagree (partial/not_covered).
    pairs = (
        [("covered", "covered")] * 6
        + [("partial", "partial")] * 2
        + [("covered", "partial")]
        + [("partial", "not_covered")]
    )
    # p_o = 8/10 = 0.8
    # rater_a: covered=7, partial=3, not_covered=0 -> 0.7, 0.3, 0.0
    # rater_b: covered=6, partial=3, not_covered=1 -> 0.6, 0.3, 0.1
    # p_e = 0.7*0.6 + 0.3*0.3 + 0.0*0.1 = 0.42 + 0.09 = 0.51
    # kappa = (0.8 - 0.51) / (1 - 0.51) = 0.29 / 0.49 ≈ 0.5918
    assert cohens_kappa(pairs) == pytest.approx(0.29 / 0.49, rel=1e-6)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# pair_by_control
# ---------------------------------------------------------------------------


def test_pair_by_control_aligns_by_id() -> None:
    det = _report([_assessment("A.5.1", "covered"), _assessment("A.5.2", "partial")])
    agt = _report(
        [_assessment("A.5.2", "not_covered"), _assessment("A.5.1", "covered")],
    )
    pairs = pair_by_control(det, agt)
    assert [(d.control_id, a.control_id) for d, a in pairs] == [
        ("A.5.1", "A.5.1"),
        ("A.5.2", "A.5.2"),
    ]


def test_pair_by_control_drops_mismatched_ids() -> None:
    det = _report([_assessment("A.5.1", "covered"), _assessment("A.5.99", "partial")])
    agt = _report([_assessment("A.5.1", "covered")])
    pairs = pair_by_control(det, agt)
    assert len(pairs) == 1 and pairs[0][0].control_id == "A.5.1"


# ---------------------------------------------------------------------------
# compare_docs
# ---------------------------------------------------------------------------


def test_compare_docs_full_agreement() -> None:
    det = _report([_assessment("A.5.1", "covered", ["c1"])])
    agt = _report([_assessment("A.5.1", "covered", ["c1"])])
    cmp = compare_docs(det, agt)
    assert cmp.total_controls == 1
    assert cmp.agreement_pct == 1.0
    assert cmp.cohens_kappa == 1.0
    assert cmp.disagreement_counts == {}
    assert cmp.evidence_jaccard_mean == 1.0


def test_compare_docs_disagreement_populated() -> None:
    det = _report(
        [
            _assessment("A.5.1", "covered"),
            _assessment("A.5.2", "covered"),
            _assessment("A.5.3", "partial"),
        ]
    )
    agt = _report(
        [
            _assessment("A.5.1", "covered"),
            _assessment("A.5.2", "partial"),  # disagrees
            _assessment("A.5.3", "not_covered"),  # disagrees
        ]
    )
    cmp = compare_docs(det, agt)
    assert cmp.total_controls == 3
    assert cmp.agreement_pct == pytest.approx(1 / 3)
    assert ("covered", "partial") in cmp.disagreement_counts
    assert ("partial", "not_covered") in cmp.disagreement_counts


def test_compare_docs_empty_reports() -> None:
    det = _report([])
    agt = _report([])
    cmp = compare_docs(det, agt)
    assert cmp.total_controls == 0
    assert cmp.agreement_pct == 0.0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_averages_per_doc_metrics() -> None:
    # Two comparisons: 100% and 50% agreement.
    det_a = _report([_assessment("A.1", "covered")], "a.pdf")
    agt_a = _report([_assessment("A.1", "covered")], "a.pdf")
    det_b = _report(
        [_assessment("A.1", "covered"), _assessment("A.2", "partial")],
        "b.pdf",
    )
    agt_b = _report(
        [_assessment("A.1", "covered"), _assessment("A.2", "not_covered")],
        "b.pdf",
    )
    agg = aggregate([compare_docs(det_a, agt_a), compare_docs(det_b, agt_b)])
    assert agg.mean_agreement_pct == pytest.approx((1.0 + 0.5) / 2)
    assert len(agg.per_doc) == 2

"""Cross-strategy comparison metrics.

All pure functions on ``Report``/``ControlAssessment`` objects — no I/O,
no MLflow. The harness computes these after both strategies have run on
a document and passes them to the MLflow logger + the Markdown writer.

Kappa is computed from a 3x3 confusion matrix without ``scikit-learn``
to keep the dep footprint small; the math is textbook and the tests
cover the edge cases (perfect agreement, chance agreement, all-disagree).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ai_auditor.models import ControlAssessment, Coverage, Report

COVERAGE_LABELS: tuple[Coverage, ...] = ("covered", "partial", "not_covered")


@dataclass(frozen=True)
class DocComparison:
    """Per-document comparison of two strategies' assessments."""

    doc_path: Path
    total_controls: int
    agreement_pct: float
    cohens_kappa: float
    disagreement_counts: dict[tuple[Coverage, Coverage], int]
    evidence_jaccard_mean: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregates across multiple document comparisons."""

    mean_agreement_pct: float
    mean_kappa: float
    mean_evidence_jaccard: float
    per_doc: list[DocComparison] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pairing + per-doc computation
# ---------------------------------------------------------------------------


def pair_by_control(det: Report, agt: Report) -> list[tuple[ControlAssessment, ControlAssessment]]:
    """Align the two reports' assessments by ``control_id``.

    Only controls present in both reports are paired; extras on either
    side are dropped silently (in practice the corpus is the same so
    this is just defensive).
    """
    agt_by_id = {a.control_id: a for a in agt.assessments}
    return [(d, agt_by_id[d.control_id]) for d in det.assessments if d.control_id in agt_by_id]


def compare_docs(det: Report, agt: Report) -> DocComparison:
    """Compare the deterministic and agentic assessments for one document."""
    pairs = pair_by_control(det, agt)
    total = len(pairs)
    if total == 0:
        return DocComparison(
            doc_path=Path(det.document_path),
            total_controls=0,
            agreement_pct=0.0,
            cohens_kappa=0.0,
            disagreement_counts={},
            evidence_jaccard_mean=0.0,
        )

    coverage_pairs = [(d.coverage, a.coverage) for d, a in pairs]
    matches = sum(1 for d, a in coverage_pairs if d == a)
    agreement_pct = matches / total
    kappa = cohens_kappa(coverage_pairs)
    disagreements = {
        pair: count for pair, count in Counter(coverage_pairs).items() if pair[0] != pair[1]
    }
    jaccards = [
        jaccard(
            {e.chunk_id for e in d.evidence},
            {e.chunk_id for e in a.evidence},
        )
        for d, a in pairs
        if d.coverage == a.coverage  # only meaningful where they agree
    ]
    evidence_jaccard_mean = sum(jaccards) / len(jaccards) if jaccards else 0.0
    return DocComparison(
        doc_path=Path(det.document_path),
        total_controls=total,
        agreement_pct=agreement_pct,
        cohens_kappa=kappa,
        disagreement_counts=disagreements,
        evidence_jaccard_mean=evidence_jaccard_mean,
    )


def aggregate(comparisons: list[DocComparison]) -> AggregateMetrics:
    """Average per-doc metrics across the comparison set."""
    if not comparisons:
        return AggregateMetrics(0.0, 0.0, 0.0, [])
    n = len(comparisons)
    return AggregateMetrics(
        mean_agreement_pct=sum(c.agreement_pct for c in comparisons) / n,
        mean_kappa=sum(c.cohens_kappa for c in comparisons) / n,
        mean_evidence_jaccard=sum(c.evidence_jaccard_mean for c in comparisons) / n,
        per_doc=list(comparisons),
    )


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity; returns 1.0 when both sets are empty."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def cohens_kappa(pairs: list[tuple[Coverage, Coverage]]) -> float:
    """Cohen's kappa over paired coverage labels.

    Computed directly from the confusion matrix: kappa = (p_o - p_e) /
    (1 - p_e). Returns 1.0 for perfect agreement, 0.0 for chance-level,
    negative for worse-than-chance. When ``p_e == 1.0`` (both raters
    use a single label), returns 1.0 iff all pairs agreed, else 0.0
    (standard convention for the undefined case).
    """
    n = len(pairs)
    if n == 0:
        return 0.0

    observed_agreement = sum(1 for a, b in pairs if a == b) / n

    # Marginal probabilities per rater per label.
    rater_a = Counter(a for a, _ in pairs)
    rater_b = Counter(b for _, b in pairs)
    p_e = sum((rater_a[label] / n) * (rater_b[label] / n) for label in COVERAGE_LABELS)

    if p_e == 1.0:
        return 1.0 if observed_agreement == 1.0 else 0.0
    return (observed_agreement - p_e) / (1.0 - p_e)

"""Shared pytest fixtures.

``sample_policy_pdf`` builds a small multi-section PDF on the fly with
reportlab. Doing it programmatically (a) keeps the repo free of binary test
fixtures, (b) lets us assert on *known* structure in tests, and (c) means the
fixture is regenerable if anything about our parser changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

SAMPLE_SECTIONS: list[tuple[str, str]] = [
    (
        "1. Access Control",
        "All employees receive role-based access to production systems. "
        "Access is provisioned by the IT team based on manager approval. "
        "Rights are reviewed every six months by the asset owner.",
    ),
    (
        "2. Incident Response",
        "The Security Incident Response Team comprises the CISO, Legal, and "
        "IT Operations leads. When an incident is detected the first "
        "responder assesses severity and escalates within thirty minutes "
        "for Major severity events.",
    ),
    (
        "3. Training",
        "All employees complete annual information security awareness "
        "training and must acknowledge the acceptable use policy on hire.",
    ),
]


@pytest.fixture(scope="session")
def sample_policy_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Produce a small three-section policy PDF for parsing / chunking tests."""
    path = tmp_path_factory.mktemp("pdf") / "sample_policy.pdf"
    _build_policy_pdf(path, SAMPLE_SECTIONS)
    return path


def _build_policy_pdf(path: Path, sections: list[tuple[str, str]]) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "Heading1Custom",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
    )
    flowables: list = []
    flowables.append(Paragraph("Example Corp — Information Security Policy", heading_style))
    flowables.append(Spacer(1, 12))
    for heading, body in sections:
        flowables.append(Paragraph(heading, heading_style))
        flowables.append(Paragraph(body, body_style))
        flowables.append(Spacer(1, 12))
    doc.build(flowables)

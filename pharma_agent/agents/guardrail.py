"""Citation guardrail: every citation in the answer must point at evidence a tool really returned.

LLMs sometimes invent ids. This check runs after the answer is written:
    * citations not in the evidence are removed and reported
    * key points left with no valid citation are counted as unsupported
    * if anything was removed, confidence drops one level
    * a not-medical-advice caveat is always present
"""

from ..schemas import Evidence, FinalAnswer, GuardrailReport, KeyPoint
from .prompts import DISCLAIMER

LOWER = {"high": "medium", "medium": "low", "low": "low"}


def check_citations(answer: FinalAnswer, evidence: list[Evidence]) -> tuple[FinalAnswer, GuardrailReport]:
    known = {e.id for e in evidence}
    invalid: list[str] = []
    valid_count = 0
    points: list[KeyPoint] = []
    unsupported = 0
    for point in answer.key_points:
        good = [c for c in point.citations if c in known]
        invalid += [c for c in point.citations if c not in known]
        valid_count += len(good)
        if not good:
            unsupported += 1
        points.append(KeyPoint(text=point.text, citations=good))

    confidence = LOWER[answer.confidence] if (invalid or unsupported) else answer.confidence
    caveats = list(answer.caveats)
    if DISCLAIMER not in caveats:
        caveats.append(DISCLAIMER)
    cleaned = FinalAnswer(answer=answer.answer, key_points=points, confidence=confidence, caveats=caveats)
    report = GuardrailReport(
        evidence_count=len(known),
        valid_citations=valid_count,
        invalid_citations=invalid,
        unsupported_points=unsupported,
    )
    return cleaned, report

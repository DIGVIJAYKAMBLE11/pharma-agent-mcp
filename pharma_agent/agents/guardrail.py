"""Citation guardrail: every citation in the answer must point at evidence a tool really returned.

LLMs sometimes invent ids, or copy a real id only partly (a live Groq run cited
"fda-label:0230a3c3" for the real id "fda-label:0230a3c3:warnings"). This check runs after
the answer is written:
    * a citation that is the start of exactly ONE real id is repaired and reported
    * any other unknown citation is removed and reported as invalid
    * key points left with no valid citation are counted as unsupported
    * if anything was removed, confidence drops one level
    * a not-medical-advice caveat is always present
"""

from ..schemas import Evidence, FinalAnswer, GuardrailReport, KeyPoint
from .prompts import DISCLAIMER

LOWER = {"high": "medium", "medium": "low", "low": "low"}


def resolve(citation: str, known: set[str]) -> str | None:
    """Exact id, or the single known id that starts with this citation; otherwise None."""
    if citation in known:
        return citation
    matches = [k for k in known if k.startswith(citation)] if len(citation) >= 8 else []
    return matches[0] if len(matches) == 1 else None


def check_citations(answer: FinalAnswer, evidence: list[Evidence]) -> tuple[FinalAnswer, GuardrailReport]:
    known = {e.id for e in evidence}
    invalid: list[str] = []
    repaired: list[str] = []
    valid_count = 0
    unsupported = 0
    points: list[KeyPoint] = []
    for point in answer.key_points:
        good: list[str] = []
        for citation in point.citations:
            fixed = resolve(citation, known)
            if fixed is None:
                invalid.append(citation)
                continue
            if fixed != citation:
                repaired.append(f"{citation} -> {fixed}")
            if fixed not in good:
                good.append(fixed)
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
        repaired_citations=repaired,
        unsupported_points=unsupported,
    )
    return cleaned, report

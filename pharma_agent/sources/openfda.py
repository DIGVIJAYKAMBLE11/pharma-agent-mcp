"""openFDA: official drug labels (prescribing information) and FAERS adverse-event counts.

API docs: https://open.fda.gov/apis/drug/label/ and https://open.fda.gov/apis/drug/event/
No API key is needed for light use.
"""

from typing import Annotated, Literal

from pydantic import Field

from ..registry import SOURCES
from ..schemas import Evidence
from .base import DataSource, clip

# Friendly section name -> label fields that hold it (first non-empty wins).
LABEL_SECTIONS: dict[str, list[str]] = {
    "indications": ["indications_and_usage"],
    "dosage": ["dosage_and_administration"],
    "contraindications": ["contraindications"],
    "warnings": ["boxed_warning", "warnings_and_cautions", "warnings"],
    "adverse_reactions": ["adverse_reactions"],
    "drug_interactions": ["drug_interactions"],
    "pregnancy": ["pregnancy", "use_in_specific_populations"],
    "mechanism": ["mechanism_of_action", "clinical_pharmacology"],
}
Section = Literal[
    "indications", "dosage", "contraindications", "warnings",
    "adverse_reactions", "drug_interactions", "pregnancy", "mechanism",
]


def _normalise_drug(drug: str) -> str:
    return drug.strip().lower().replace('"', "")


@SOURCES.register("openfda_labels")
class OpenFDALabels(DataSource):
    name = "search_drug_labels"
    description = (
        "Read one section of the official FDA drug label (prescribing information) for a drug. "
        "Use for indications, dosage, contraindications, warnings (incl. boxed warning), "
        "adverse reactions, drug interactions, pregnancy, or mechanism of action."
    )
    URL = "https://api.fda.gov/drug/label.json"
    MAX_CHARS = 1500

    def search(
        self,
        drug: Annotated[str, Field(description="Generic or brand name, e.g. 'metformin' or 'Eliquis'")],
        section: Annotated[Section, Field(description="Which label section to return")] = "drug_interactions",
        limit: Annotated[int, Field(ge=1, le=3, description="How many labels to return")] = 1,
    ) -> list[Evidence]:
        name = _normalise_drug(drug)
        query = f'openfda.generic_name:"{name}" OR openfda.brand_name:"{name}"'
        res = self.http.get(self.URL, {"search": query, "limit": 20})
        if res.status == 404:
            return []
        fields = LABEL_SECTIONS[section]
        labels = [lab for lab in res.json().get("results", []) if any(lab.get(f) for f in fields)]
        # Prefer single-ingredient products, then the most recently updated label.
        labels.sort(key=lambda lab: (
            len(lab.get("openfda", {}).get("substance_name") or ["?"] * 9),
            -int(lab.get("effective_time") or 0),
        ))
        return [self._to_evidence(lab, section, fields) for lab in labels[:limit]]

    def _to_evidence(self, label: dict, section: str, fields: list[str]) -> Evidence:
        meta = label.get("openfda", {})
        brand = ", ".join(meta.get("brand_name", [])[:2]) or "unknown brand"
        generic = ", ".join(meta.get("generic_name", [])[:1]) or "unknown generic"
        field = next(f for f in fields if label.get(f))
        set_id = label.get("set_id", label.get("id", "unknown"))
        return Evidence(
            id=f"fda-label:{set_id}:{section}",
            source="openfda_labels",
            title=f"FDA label: {brand} ({generic}), {section.replace('_', ' ')}",
            text=clip(" ".join(label[field]), self.MAX_CHARS),
            url=f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}",
            metadata={
                "label_field": field,
                "effective_time": label.get("effective_time"),
                "manufacturer": (meta.get("manufacturer_name") or [None])[0],
            },
        )


@SOURCES.register("openfda_events")
class OpenFDAAdverseEvents(DataSource):
    name = "get_adverse_event_counts"
    description = (
        "Count the most frequently reported adverse reactions for a drug in the FDA Adverse Event "
        "Reporting System (FAERS). Counts are spontaneous reports, not proof that the drug caused them."
    )
    URL = "https://api.fda.gov/drug/event.json"

    def search(
        self,
        drug: Annotated[str, Field(description="Generic drug name, e.g. 'atorvastatin'")],
        top_n: Annotated[int, Field(ge=1, le=25, description="How many reactions to return")] = 10,
    ) -> list[Evidence]:
        name = _normalise_drug(drug)
        params = {
            "search": f'patient.drug.openfda.generic_name:"{name}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": top_n,
        }
        res = self.http.get(self.URL, params)
        if res.status == 404:
            return []
        terms = res.json().get("results", [])
        if not terms:
            return []
        text = "; ".join(f"{t['term'].capitalize()}: {t['count']:,} reports" for t in terms)
        return [
            Evidence(
                id=f"faers:{name.replace(' ', '-')}",
                source="openfda_events",
                title=f"FAERS top {len(terms)} reported reactions for {name}",
                text=f"Most reported adverse reactions (FAERS, spontaneous reports, not causal): {text}",
                url=str(self.http.cache_key(self.URL, params)),
                metadata={"top_terms": [t["term"] for t in terms]},
            )
        ]

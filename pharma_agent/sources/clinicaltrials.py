"""ClinicalTrials.gov API v2: search registered clinical studies.

API docs: https://clinicaltrials.gov/data-api/api . No key needed.
"""

from typing import Annotated, Literal

from pydantic import Field

from ..registry import SOURCES
from ..schemas import Evidence
from .base import DataSource, clip

Status = Literal["ANY", "RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED"]


@SOURCES.register("clinicaltrials")
class ClinicalTrials(DataSource):
    name = "search_clinical_trials"
    description = (
        "Search ClinicalTrials.gov for registered studies by condition and/or intervention (drug), "
        "optionally filtered by recruitment status. Returns NCT id, title, phase, status, "
        "enrollment, sponsor and a short summary."
    )
    URL = "https://clinicaltrials.gov/api/v2/studies"
    FIELDS = ",".join([
        "NCTId", "BriefTitle", "OverallStatus", "Phase", "Condition", "InterventionName",
        "BriefSummary", "EnrollmentCount", "StartDate", "LeadSponsorName",
    ])

    def search(
        self,
        condition: Annotated[str, Field(description="Disease or condition, e.g. 'type 2 diabetes'")] = "",
        intervention: Annotated[str, Field(description="Drug or other intervention, e.g. 'semaglutide'")] = "",
        status: Annotated[Status, Field(description="Recruitment status filter")] = "ANY",
        limit: Annotated[int, Field(ge=1, le=10, description="How many studies to return")] = 5,
    ) -> list[Evidence]:
        if not condition.strip() and not intervention.strip():
            raise ValueError("Give at least a condition or an intervention.")
        params: dict = {"pageSize": limit, "fields": self.FIELDS, "format": "json"}
        if condition.strip():
            params["query.cond"] = condition.strip()
        if intervention.strip():
            params["query.intr"] = intervention.strip()
        if status != "ANY":
            params["filter.overallStatus"] = status
        studies = self.http.get(self.URL, params).json().get("studies", [])
        return [self._to_evidence(s.get("protocolSection", {})) for s in studies]

    @staticmethod
    def _to_evidence(p: dict) -> Evidence:
        ident = p.get("identificationModule", {})
        status = p.get("statusModule", {})
        design = p.get("designModule", {})
        nct = ident.get("nctId", "unknown")
        phases = ", ".join(design.get("phases", [])) or "n/a"
        conditions = ", ".join(p.get("conditionsModule", {}).get("conditions", [])[:4])
        interventions = ", ".join(i.get("name", "") for i in p.get("armsInterventionsModule", {}).get("interventions", [])[:4])
        enrollment = design.get("enrollmentInfo", {}).get("count", "n/a")
        sponsor = p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "n/a")
        start = status.get("startDateStruct", {}).get("date", "n/a")
        summary = clip(p.get("descriptionModule", {}).get("briefSummary", ""), 500)
        return Evidence(
            id=f"nct:{nct}",
            source="clinicaltrials",
            title=f"{nct}: {ident.get('briefTitle', '')}",
            text=(
                f"Status: {status.get('overallStatus', 'n/a')}; Phase: {phases}; Conditions: {conditions}; "
                f"Interventions: {interventions}; Enrollment: {enrollment}; Sponsor: {sponsor}; "
                f"Start: {start}. Summary: {summary}"
            ),
            url=f"https://clinicaltrials.gov/study/{nct}",
            metadata={"status": status.get("overallStatus"), "phases": design.get("phases", [])},
        )

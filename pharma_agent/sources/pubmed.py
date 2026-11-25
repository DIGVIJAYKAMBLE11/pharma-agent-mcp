"""PubMed via NCBI E-utilities: search biomedical literature and read abstracts.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/ . No key needed for light use
(NCBI allows about 3 requests per second without one).
"""

import xml.etree.ElementTree as ET
from typing import Annotated

from pydantic import Field

from ..registry import SOURCES
from ..schemas import Evidence
from .base import DataSource, clip


@SOURCES.register("pubmed")
class PubMed(DataSource):
    name = "search_pubmed"
    description = (
        "Search PubMed biomedical literature. Returns the most relevant articles with title, "
        "journal, year and abstract. Use for mechanisms, comparative evidence, reviews and recent research."
    )
    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    MAX_CHARS = 900

    def search(
        self,
        query: Annotated[str, Field(description="PubMed search terms, e.g. 'metformin lactic acidosis risk'")],
        limit: Annotated[int, Field(ge=1, le=5, description="How many articles to return")] = 3,
    ) -> list[Evidence]:
        found = self.http.get(
            self.ESEARCH,
            {"db": "pubmed", "term": query, "retmode": "json", "retmax": limit, "sort": "relevance"},
        ).json()
        ids = found.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        xml = self.http.get(self.EFETCH, {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}).text
        articles = {a["pmid"]: a for a in self._parse(xml)}
        return [self._to_evidence(articles[i]) for i in ids if i in articles]

    @staticmethod
    def _parse(xml: str) -> list[dict]:
        out = []
        for art in ET.fromstring(xml).iter("PubmedArticle"):
            title_el = art.find(".//ArticleTitle")
            date = art.find(".//JournalIssue/PubDate")
            year = (date.findtext("Year") or date.findtext("MedlineDate") or "")[:4] if date is not None else ""
            out.append({
                "pmid": art.findtext(".//MedlineCitation/PMID", default=""),
                "title": "".join(title_el.itertext()) if title_el is not None else "",
                "abstract": " ".join("".join(a.itertext()) for a in art.iter("AbstractText")),
                "journal": art.findtext(".//Journal/Title", default=""),
                "year": year,
            })
        return out

    def _to_evidence(self, a: dict) -> Evidence:
        return Evidence(
            id=f"pmid:{a['pmid']}",
            source="pubmed",
            title=f"{a['title']} ({a['journal']}, {a['year']})",
            text=clip(a["abstract"] or "No abstract available.", self.MAX_CHARS),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/",
            metadata={"journal": a["journal"], "year": a["year"]},
        )

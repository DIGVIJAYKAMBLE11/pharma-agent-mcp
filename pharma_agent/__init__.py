"""PharmaAgent: a multi-agent assistant over public drug and clinical-trial data.

Layers, from the bottom up (each layer only knows the one below it):

    sources/   fetch and normalise public data into `Evidence`   (openFDA, ClinicalTrials.gov, PubMed)
    tools.py   turn each source into a tool function             (one function, two transports)
    mcp_server serve those tools over the Model Context Protocol
    agents/    LangGraph supervisor -> specialist agents -> cited, structured answer
    service.py one entry point: `await PharmaAgent.create()` then `await agent.ask(question)`
    api.py     FastAPI wrapper around the service
"""

__version__ = "0.1.0"

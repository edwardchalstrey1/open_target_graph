from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

# 1. Define the Output Structure (Pydantic)
# This ensures structured data for the dashboard.
class PaperSummary(BaseModel):
    title: str
    authors: List[str]
    year: int
    abstract_summary: str
    key_findings: List[str]
    relevance_score: int = Field(description="1-10 score of relevance to drug discovery")
    pubmed_id: str

class TargetReport(BaseModel):
    target_name: str
    uniprot_id: str
    mechanism_summary: str
    top_papers: List[PaperSummary]
    clinical_trial_status: Optional[str] = Field(description="Current known clinical trial phase if any")
    recommendation: str = Field(description="Go/No-Go recommendation for further research")
    generated_at: str

# 2. Define the Agent's "System Prompt"
SYSTEM_PROMPT = """
You are a Senior Computational Biologist and Drug Discovery Expert.
Your goal is to evaluate protein targets for drug discovery potential based on the latest scientific literature.

You will be provided with a protein name and its UniProt ID. 
You must:
1. Search for recent (last 10 years) high-impact papers on PubMed about this target's role in disease and its druggability.
2. Analyze and summarize the papers.
3. Provide a structured report including a "Go/No-Go" recommendation.

Always return structured results that match the TargetReport schema.
Be objective, cite your sources via PubMed IDs, and highlight any clinical relevance.
"""

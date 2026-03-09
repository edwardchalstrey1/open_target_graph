import os
import requests
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from datetime import datetime
from google import genai
from google.genai import types
from open_target_graph.agents.researcher import TargetReport, PaperSummary, SYSTEM_PROMPT

def get_client():
    """Returns a configured Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment.")
    return genai.Client(api_key=api_key)

# Define the "State" of our agent
class AgentState(TypedDict):
    uniprot_id: str
    protein_name: str
    model_id: str
    query: str
    raw_papers: List[Dict[str, Any]]
    final_report: Dict[str, Any]
    error: str

# --- PubMed Tools ---

def search_pubmed(state: AgentState) -> Dict[str, Any]:
    """Search PubMed for the given target."""
    target_name = state["protein_name"]
    query = f"{target_name} drug target discovery"
    
    # PubMed E-search API
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 5,
        "sort": "relevance"
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        id_list = response.json().get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return {"raw_papers": [], "error": "No papers found on PubMed."}
        
        # PubMed E-summary API
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json"
        }
        
        summary_response = requests.get(summary_url, params=summary_params)
        summary_response.raise_for_status()
        summaries = summary_response.json().get("result", {})
        
        raw_papers = []
        for pm_id in id_list:
            paper_data = summaries.get(pm_id, {})
            raw_papers.append({
                "pubmed_id": pm_id,
                "title": paper_data.get("title", ""),
                "authors": [a.get("name", "") for a in paper_data.get("authors", [])],
                "pubdate": paper_data.get("pubdate", ""),
                "fulljournalname": paper_data.get("fulljournalname", "")
            })
            
        return {"raw_papers": raw_papers}
        
    except Exception as e:
        return {"error": f"PubMed API error: {str(e)}"}

def analyze_papers(state: AgentState) -> Dict[str, Any]:
    """Use an LLM to analyze the found papers and generate a report."""
    if state.get("error"):
        return {}

    model_id = state.get("model_id", "gemini-flash-latest")
    
    prompt = f"""
    Analyze the following papers for the target {state['protein_name']} ({state['uniprot_id']}):
    {state['raw_papers']}
    
    Generate a research report following this JSON structure exactly:
    {{
        "target_name": "{state['protein_name']}",
        "uniprot_id": "{state['uniprot_id']}",
        "mechanism_summary": "...",
        "top_papers": [
            {{
                "title": "...",
                "authors": ["..."],
                "year": 2024,
                "abstract_summary": "...",
                "key_findings": ["..."],
                "relevance_score": 9,
                "pubmed_id": "..."
            }}
        ],
        "clinical_trial_status": "...",
        "recommendation": "...",
        "generated_at": "{datetime.now().isoformat()}"
    }}
    """
    
    try:
        client = get_client()
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type='application/json'
            )
        )
        
        import json
        report_data = json.loads(response.text)
        return {"final_report": report_data}
            
    except Exception as e:
        return {"error": f"LLM error: {str(e)}"}

# --- Graph Definition ---

def create_research_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("search", search_pubmed)
    workflow.add_node("analyze", analyze_papers)
    
    workflow.set_entry_point("search")
    workflow.add_edge("search", "analyze")
    workflow.add_edge("analyze", END)
    
    return workflow.compile()

research_app = create_research_workflow()

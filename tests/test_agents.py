import pytest
import requests
from open_target_graph.agents.workflow import search_pubmed, analyze_papers, AgentState, research_app
from open_target_graph.agents.researcher import SYSTEM_PROMPT

class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(f"Status {self.status_code}")

def test_search_pubmed_success(mocker):
    """Tests that search_pubmed correctly fetches and parses papers."""
    mock_search_response = MockResponse({
        "esearchresult": {"idlist": ["123", "456"]}
    }, 200)
    
    mock_summary_response = MockResponse({
        "result": {
            "123": {"title": "Paper 1", "authors": [{"name": "Auth 1"}], "pubdate": "2024", "fulljournalname": "Journal 1"},
            "456": {"title": "Paper 2", "authors": [{"name": "Auth 2"}], "pubdate": "2023", "fulljournalname": "Journal 2"}
        }
    }, 200)
    
    mocker.patch("requests.get", side_effect=[mock_search_response, mock_summary_response])
    
    state: AgentState = {
        "uniprot_id": "P12345",
        "protein_name": "TestProt",
        "model_id": "gemini-flash",
        "query": "",
        "raw_papers": [],
        "final_report": {},
        "error": ""
    }
    
    result = search_pubmed(state)
    assert "raw_papers" in result
    assert len(result["raw_papers"]) == 2
    assert result["raw_papers"][0]["title"] == "Paper 1"
    assert result["raw_papers"][1]["pubmed_id"] == "456"

def test_search_pubmed_no_results(mocker):
    """Tests search_pubmed when no papers are found."""
    mock_search_response = MockResponse({
        "esearchresult": {"idlist": []}
    }, 200)
    
    mocker.patch("requests.get", return_value=mock_search_response)
    
    state: AgentState = {
        "uniprot_id": "P12345",
        "protein_name": "TestProt",
        "model_id": "gemini-flash",
        "query": "",
        "raw_papers": [],
        "final_report": {},
        "error": ""
    }
    
    result = search_pubmed(state)
    assert result["raw_papers"] == []
    assert "error" in result

def test_analyze_papers_success(mocker):
    """Tests that analyze_papers correctly calls Gemini and returns a report."""
    mock_client = mocker.Mock()
    mock_response = mocker.Mock()
    mock_response.text = '{"target_name": "TestProt", "mechanism_summary": "Test mechanism"}'
    mock_client.models.generate_content.return_value = mock_response
    
    mocker.patch("open_target_graph.agents.workflow.get_client", return_value=mock_client)
    
    state: AgentState = {
        "uniprot_id": "P12345",
        "protein_name": "TestProt",
        "model_id": "gemini-flash",
        "query": "",
        "raw_papers": [{"title": "Paper 1"}],
        "final_report": {},
        "error": ""
    }
    
    result = analyze_papers(state)
    assert "final_report" in result
    assert result["final_report"]["target_name"] == "TestProt"

def test_analyze_papers_error(mocker):
    """Tests analyze_papers when Gemini fails."""
    mocker.patch("open_target_graph.agents.workflow.get_client", side_effect=Exception("LLM Error"))
    
    state: AgentState = {
        "uniprot_id": "P12345",
        "protein_name": "TestProt",
        "model_id": "gemini-flash",
        "query": "",
        "raw_papers": [{"title": "Paper 1"}],
        "final_report": {},
        "error": ""
    }
    
    result = analyze_papers(state)
    assert "error" in result
    assert "LLM error" in result["error"]

def test_research_workflow_structure():
    """Verifies the structure of the LangGraph workflow."""
    # Ensure it's compiled correctly
    assert hasattr(research_app, "invoke")
    # We can't easily test the graph edges/nodes without deeper inspection, 
    # but the fact it compile and has an entry point is good.

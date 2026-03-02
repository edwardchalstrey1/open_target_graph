import pytest
import requests
import numpy as np
import polars as pl

from open_target_graph.dashboard.app import fetch_pdb_data, compute_tsne_projection, find_similar_targets


class MockResponse:
    def __init__(self, json_data, status_code, text_data=""):
        self.json_data = json_data
        self.status_code = status_code
        self.text = text_data

    def json(self):
        return self.json_data

    @property
    def ok(self):
        return self.status_code == 200


def test_fetch_pdb_data_success(mocker):
    """Tests successful PDB data fetching."""
    mock_api_response = MockResponse([{"pdbUrl": "http://example.com/pdb"}], 200)
    mock_pdb_response = MockResponse(None, 200, text_data="PDB_CONTENT")
    mocker.patch("requests.get", side_effect=[mock_api_response, mock_pdb_response])

    result = fetch_pdb_data("P12345")
    assert result == "PDB_CONTENT"


def test_fetch_pdb_data_api_not_found(mocker):
    """Tests when the AlphaFold API returns no entry."""
    mock_api_response = MockResponse([], 200)
    mocker.patch("requests.get", return_value=mock_api_response)

    result = fetch_pdb_data("P12345")
    assert result is None


def test_fetch_pdb_data_request_exception(mocker):
    """Tests when a network error occurs."""
    mocker.patch("requests.get", side_effect=requests.RequestException)

    result = fetch_pdb_data("P12345")
    assert result is None


def test_compute_tsne_projection():
    """Tests that t-SNE projection returns the correct shape."""
    # Create 10 dummy embeddings of dimension 50
    dummy_embeddings = np.random.rand(10, 50).tolist()

    projections = compute_tsne_projection(dummy_embeddings, perplexity=5)

    assert isinstance(projections, np.ndarray)
    # Should return 10 points, each with 2 coordinates (x, y)
    assert projections.shape == (10, 2)


def test_compute_tsne_projection_empty_input():
    """Tests t-SNE with empty input."""
    with pytest.raises(ValueError):
        compute_tsne_projection([])


def test_find_similar_targets():
    """Tests cosine similarity search logic."""
    # Create dummy data
    # Target: [1.0, 0.0]
    # Match:  [0.99, 0.0] (Same direction, different magnitude -> Sim = 1.0)
    # Ortho:  [0.0, 1.0]  (Orthogonal -> Sim = 0.0)
    # Oppos:  [-1.0, 0.0] (Opposite -> Sim = -1.0)
    
    data = {
        "uniprot_id": ["TGT", "MATCH", "ORTHO", "OPPOS"],
        "protein_name": ["Target", "Match", "Ortho", "Oppos"],
        "gene_name": ["T", "M", "O", "O"],
        "embedding": [
            [1.0, 0.0],
            [0.99, 0.0], 
            [0.0, 1.0],
            [-1.0, 0.0]
        ]
    }
    df = pl.DataFrame(data)
    
    # Find similar to "TGT", top_n=2
    # Should filter out TGT itself.
    # Expected order: MATCH (sim=1.0), ORTHO (sim=0.0), OPPOS (sim=-1.0)
    results = find_similar_targets(df, "TGT", top_n=2)
    
    assert len(results) == 2
    assert results["uniprot_id"][0] == "MATCH"
    assert results["uniprot_id"][1] == "ORTHO"
    
    # Check similarity values
    # MATCH: (1*0.99 + 0*0) / (1 * 0.99) = 1.0
    assert results["similarity"][0] == pytest.approx(1.0, rel=1e-3)
    # ORTHO: 0.0
    assert results["similarity"][1] == pytest.approx(0.0, abs=1e-3)
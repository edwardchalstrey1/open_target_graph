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


class MockSessionState(dict):
    def __getattr__(self, name): return self.get(name)
    def __setattr__(self, name, val): self[name] = val
    def __delattr__(self, name):
        if name in self: del self[name]


def test_configure_page(mocker):
    mock_set_page_config = mocker.patch("open_target_graph.dashboard.app.st.set_page_config")
    from open_target_graph.dashboard.app import configure_page
    configure_page()
    mock_set_page_config.assert_called_once_with(layout="wide", page_title="OpenTargetGraph")


def test_render_header(mocker):
    mock_title = mocker.patch("open_target_graph.dashboard.app.st.title")
    mock_markdown = mocker.patch("open_target_graph.dashboard.app.st.markdown")
    from open_target_graph.dashboard.app import render_header
    render_header()
    mock_title.assert_called_once()
    mock_markdown.assert_called_once()


def test_load_data_success(mocker):
    mock_read_parquet = mocker.patch("open_target_graph.dashboard.app.pl.read_parquet")
    df1 = pl.DataFrame({"uniprot_id": ["A"], "val": [1]})
    df2 = pl.DataFrame({"uniprot_id": ["A"], "embedding": [[1.0, 2.0]]})
    mock_read_parquet.side_effect = [df1, df2]
    
    from open_target_graph.dashboard.app import load_data
    load_data.clear() # clear cache
    result = load_data()
    
    assert "val" in result.columns
    assert "embedding" in result.columns
    assert len(result) == 1


def test_load_data_error(mocker):
    mocker.patch("open_target_graph.dashboard.app.pl.read_parquet", side_effect=Exception("Test Error"))
    mock_error = mocker.patch("open_target_graph.dashboard.app.st.error")
    mock_stop = mocker.patch("open_target_graph.dashboard.app.st.stop")
    
    from open_target_graph.dashboard.app import load_data
    load_data.clear()
    load_data()
    
    mock_error.assert_called_once()
    mock_stop.assert_called_once()


def test_load_chembl_data_success(mocker):
    mock_read_parquet = mocker.patch("open_target_graph.dashboard.app.pl.read_parquet")
    df = pl.DataFrame({"test": [1]})
    mock_read_parquet.return_value = df
    
    from open_target_graph.dashboard.app import load_chembl_data
    load_chembl_data.clear()
    result = load_chembl_data()
    
    assert result.equals(df)


def test_load_chembl_data_empty(mocker):
    mock_read_parquet = mocker.patch("open_target_graph.dashboard.app.pl.read_parquet")
    mock_read_parquet.return_value = pl.DataFrame()
    mock_warning = mocker.patch("open_target_graph.dashboard.app.st.warning")
    
    from open_target_graph.dashboard.app import load_chembl_data
    load_chembl_data.clear()
    result = load_chembl_data()
    
    mock_warning.assert_called_once()
    assert result.is_empty()


def test_load_chembl_data_exception(mocker):
    mocker.patch("open_target_graph.dashboard.app.pl.read_parquet", side_effect=Exception("Test Exception"))
    mock_warning = mocker.patch("open_target_graph.dashboard.app.st.warning")
    
    from open_target_graph.dashboard.app import load_chembl_data
    load_chembl_data.clear()
    result = load_chembl_data()
    
    mock_warning.assert_called_once()
    assert result.is_empty()


def test_create_3d_view(mocker):
    mock_view_class = mocker.patch("open_target_graph.dashboard.app.py3Dmol.view")
    mock_view_instance = mock_view_class.return_value
    
    from open_target_graph.dashboard.app import create_3d_view
    view = create_3d_view("TEST_PDB", width=100, height=100)
    
    mock_view_class.assert_called_once_with(width=100, height=100)
    mock_view_instance.addModel.assert_called_once_with("TEST_PDB", "pdb")
    mock_view_instance.setStyle.assert_called_once_with({'cartoon': {'color': 'spectrum'}})
    mock_view_instance.zoomTo.assert_called_once()
    assert view == mock_view_instance


def test_render_target_selection(mocker):
    mocker.patch("open_target_graph.dashboard.app.st.subheader")
    mock_st = mocker.patch("open_target_graph.dashboard.app.st")
    mock_st.session_state = MockSessionState({"selected_id": "A"})
    
    df = pl.DataFrame({
        "uniprot_id": ["A", "B"],
        "protein_name": ["Protein A", "Protein B"],
        "sequence": ["SEQ_A", "SEQ_B"]
    })
    
    from open_target_graph.dashboard.app import render_target_selection
    render_target_selection(df)
    
    mock_st.selectbox.assert_called_once()
    mock_st.text_area.assert_called_once_with("Sequence", "SEQ_A", height=100)


def test_render_structure_preview_found(mocker):
    mocker.patch("open_target_graph.dashboard.app.st.subheader")
    mock_fetch_pdb = mocker.patch("open_target_graph.dashboard.app.fetch_pdb_data", return_value="PDB_DATA")
    mock_create_view = mocker.patch("open_target_graph.dashboard.app.create_3d_view", return_value="MOCK_VIEW")
    mock_showmol = mocker.patch("open_target_graph.dashboard.app.showmol")
    
    from open_target_graph.dashboard.app import render_structure_preview
    render_structure_preview("A")
    
    mock_fetch_pdb.assert_called_once_with("A")
    mock_create_view.assert_called_once_with("PDB_DATA")
    mock_showmol.assert_called_once_with("MOCK_VIEW", height=300, width=400)


def test_render_structure_preview_not_found(mocker):
    mocker.patch("open_target_graph.dashboard.app.st.subheader")
    mock_warning = mocker.patch("open_target_graph.dashboard.app.st.warning")
    mock_fetch_pdb = mocker.patch("open_target_graph.dashboard.app.fetch_pdb_data", return_value=None)
    mock_view_class = mocker.patch("open_target_graph.dashboard.app.py3Dmol.view")
    mock_showmol = mocker.patch("open_target_graph.dashboard.app.showmol")
    
    from open_target_graph.dashboard.app import render_structure_preview
    render_structure_preview("A")
    
    mock_warning.assert_called_once()
    mock_view_class.assert_called_once()
    mock_showmol.assert_called_once()


def test_render_similarity_search(mocker):
    mock_find_similar = mocker.patch("open_target_graph.dashboard.app.find_similar_targets")
    ret_df = pl.DataFrame({"uniprot_id": ["B"], "protein_name": ["P"], "gene_name": ["G"], "similarity": [0.9]})
    mock_find_similar.return_value = ret_df
    
    mocker.patch("open_target_graph.dashboard.app.st.markdown")
    mock_dataframe = mocker.patch("open_target_graph.dashboard.app.st.dataframe")
    
    from open_target_graph.dashboard.app import render_similarity_search
    df = pl.DataFrame()
    result = render_similarity_search(df, "A")
    
    assert result.equals(ret_df)
    mock_dataframe.assert_called_once()


def test_render_drug_candidates_empty_chembl(mocker):
    mock_st_subheader = mocker.patch("open_target_graph.dashboard.app.st.subheader")
    chembl_df = pl.DataFrame()
    similar_targets_df = pl.DataFrame()
    
    from open_target_graph.dashboard.app import render_drug_candidates
    render_drug_candidates(chembl_df, similar_targets_df, "A")
    
    mock_st_subheader.assert_called_once()


def test_render_drug_candidates_no_matches(mocker):
    mock_st_info = mocker.patch("open_target_graph.dashboard.app.st.info")
    chembl_df = pl.DataFrame({"uniprot_id": ["C"], "pchembl_value": [5.0]})
    similar_targets_df = pl.DataFrame({"uniprot_id": ["B"]})
    
    from open_target_graph.dashboard.app import render_drug_candidates
    render_drug_candidates(chembl_df, similar_targets_df, "A")
    
    mock_st_info.assert_called_once()


def test_render_drug_candidates_with_matches(mocker):
    mock_st_dataframe = mocker.patch("open_target_graph.dashboard.app.st.dataframe")
    chembl_df = pl.DataFrame({
        "uniprot_id": ["A", "B"], 
        "pchembl_value": [5.0, 6.0], 
        "pref_name": ["Drug1", "Drug2"], 
        "standard_type": ["IC50", "IC50"]
    })
    similar_targets_df = pl.DataFrame({"uniprot_id": ["B"]})
    
    from open_target_graph.dashboard.app import render_drug_candidates
    render_drug_candidates(chembl_df, similar_targets_df, "A")
    
    mock_st_dataframe.assert_called_once()
    display_df = mock_st_dataframe.call_args[0][0]
    assert "Candidate Type" in display_df.columns


def test_render_tsne_plot(mocker):
    mock_compute = mocker.patch("open_target_graph.dashboard.app.compute_tsne_projection", return_value=np.array([[0, 0], [1, 1]]))
    mock_px_scatter = mocker.patch("open_target_graph.dashboard.app.px.scatter")
    mock_fig = mocker.MagicMock()
    mock_px_scatter.return_value = mock_fig
    
    mock_st = mocker.patch("open_target_graph.dashboard.app.st")
    mock_st.session_state = MockSessionState({"selected_id": "A"})
    mock_st.plotly_chart.return_value = None
    
    df = pl.DataFrame({
        "uniprot_id": ["A", "B"],
        "embedding": [[0.0, 0.0], [1.0, 1.0]],
        "protein_name": ["PA", "PB"],
        "gene_name": ["GA", "GB"],
        "length": [10, 20]
    })
    
    from open_target_graph.dashboard.app import render_tsne_plot
    render_tsne_plot(df)
    
    mock_compute.assert_called_once()
    mock_px_scatter.assert_called_once()
    mock_st.plotly_chart.assert_called_once()
    mock_fig.add_scatter.assert_called_once()


def test_main_function(mocker):
    mock_configure = mocker.patch("open_target_graph.dashboard.app.configure_page")
    mock_header = mocker.patch("open_target_graph.dashboard.app.render_header")
    mock_load_data = mocker.patch("open_target_graph.dashboard.app.load_data", return_value=pl.DataFrame({"uniprot_id": ["A"]}))
    mock_load_chembl = mocker.patch("open_target_graph.dashboard.app.load_chembl_data", return_value=pl.DataFrame())
    mock_target_sel = mocker.patch("open_target_graph.dashboard.app.render_target_selection")
    mock_sim_search = mocker.patch("open_target_graph.dashboard.app.render_similarity_search", return_value=pl.DataFrame())
    mock_struct_prev = mocker.patch("open_target_graph.dashboard.app.render_structure_preview")
    mock_tsne = mocker.patch("open_target_graph.dashboard.app.render_tsne_plot")
    mock_drug_cand = mocker.patch("open_target_graph.dashboard.app.render_drug_candidates")
    
    mock_st = mocker.patch("open_target_graph.dashboard.app.st")
    mock_st.session_state = MockSessionState({})
    mock_st.columns.return_value = (mocker.MagicMock(), mocker.MagicMock())
    
    from open_target_graph.dashboard.app import main
    main()
    
    mock_configure.assert_called_once()
    mock_header.assert_called_once()
    mock_load_data.assert_called_once()
    mock_load_chembl.assert_called_once()
    mock_target_sel.assert_called_once()
    mock_sim_search.assert_called_once()
    mock_struct_prev.assert_called_once()
    mock_tsne.assert_called_once()
    mock_drug_cand.assert_called_once()

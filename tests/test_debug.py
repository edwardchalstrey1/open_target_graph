import pytest
import polars as pl
from unittest.mock import MagicMock
from open_target_graph.dashboard.app import load_data
import streamlit as st

def test_load_data_debug(mocker):
    mock_create_engine = mocker.patch("open_target_graph.dashboard.app.create_engine")
    mock_read_db = mocker.patch("open_target_graph.dashboard.app.pl.read_database")
    mock_error = mocker.patch("open_target_graph.dashboard.app.st.error")
    mock_stop = mocker.patch("open_target_graph.dashboard.app.st.stop")
    
    df = pl.DataFrame({
        "uniprot_id": ["A"],
        "val": [1],
        "embedding_str": ["[1.0, 2.0]"]
    })
    mock_read_db.return_value = df
    
    load_data.clear()
    
    # We want to see what st.error was called with
    def print_error(*args, **kwargs):
        print(f"ST.ERROR CALLED WITH {args}")
        
    mock_error.side_effect = print_error
    
    result = load_data()
    print("RESULT IS", result)

if __name__ == "__main__":
    pytest.main(["-s", "test_debug.py"])

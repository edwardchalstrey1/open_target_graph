import pytest
import polars as pl
import torch
from dagster import build_asset_context
from unittest.mock import MagicMock

from open_target_graph.assets.modeling.inference import protein_embeddings, MODEL_NAME


@pytest.fixture
def mock_uniprot_parquet(tmp_path) -> str:
    """Creates a dummy parquet file for testing the inference asset."""
    df = pl.DataFrame({
        "uniprot_id": ["P12345", "P67890"],
        "sequence": ["MKT...Y", "MAV...K"]
    })
    path = tmp_path / "kinases.parquet"
    df.write_parquet(path)
    return str(path)


def test_protein_embeddings(mocker, mock_uniprot_parquet):
    """Tests the protein_embeddings asset logic."""
    # 1. Mock the HuggingFace transformers library
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.ones(2, 10),
        "attention_mask": torch.ones(2, 10)
    }
    mocker.patch("open_target_graph.assets.modeling.inference.AutoTokenizer.from_pretrained", return_value=mock_tokenizer)

    mock_model_output = MagicMock()
    # Mock output: Batch x Seq_Length x Vector_Size (320 for esm2_t6_8M)
    mock_model_output.last_hidden_state = torch.rand(2, 10, 320)
    mock_model = MagicMock()
    mock_model.return_value = mock_model_output
    mocker.patch("open_target_graph.assets.modeling.inference.AutoModel.from_pretrained", return_value=mock_model)

    # 2. Mock the output file write to inspect the DataFrame
    mock_write_parquet = mocker.patch("polars.DataFrame.write_parquet")

    # 3. Build context and run the asset
    context = build_asset_context()
    result_path = protein_embeddings(context, uniprot_parquet=mock_uniprot_parquet)

    # 4. Assertions
    assert result_path == "data/embeddings.parquet"

    # Check that the model and tokenizer were loaded with the correct name
    mocker.patch("open_target_graph.assets.modeling.inference.AutoTokenizer.from_pretrained").assert_called_with(MODEL_NAME)
    mocker.patch("open_target_graph.assets.modeling.inference.AutoModel.from_pretrained").assert_called_with(MODEL_NAME)

    # Check that the output DataFrame is correct
    mock_write_parquet.assert_called_once()
    call_args = mock_write_parquet.call_args
    output_df = call_args[0][0]  # The DataFrame is the first positional arg

    assert isinstance(output_df, pl.DataFrame)
    assert output_df.shape == (2, 2)
    assert output_df.columns == ["uniprot_id", "embedding"]
    assert output_df["uniprot_id"].to_list() == ["P12345", "P67890"]
    assert len(output_df["embedding"][0]) == 320  # Check embedding dimension
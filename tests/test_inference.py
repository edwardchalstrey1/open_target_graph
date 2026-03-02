import torch
from dagster import build_asset_context
from unittest.mock import MagicMock

from open_target_graph.assets.modeling.inference import (
    protein_embeddings, 
    generate_embeddings, 
    MODEL_NAME
)


def test_generate_embeddings():
    """Unit test for the embedding generation logic."""
    # Mock tokenizer
    mock_tokenizer = MagicMock()
    # Return a dict with tensors. The shape doesn't strictly matter for this test's logic.
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 5), # Simulating batch size of 1
        "attention_mask": torch.zeros(1, 5)
    }
    
    # Mock model
    mock_model = MagicMock()

    # Create two distinct outputs for the two expected calls
    mock_output_1 = MagicMock()
    # Batch=1, Seq=5, Dim=3. Mean will be [1.0, 1.0, 1.0]
    mock_output_1.last_hidden_state = torch.tensor([[[1.0, 1.0, 1.0]] * 5])

    mock_output_2 = MagicMock()
    # Batch=1, Seq=5, Dim=3. Mean will be [2.0, 2.0, 2.0]
    mock_output_2.last_hidden_state = torch.tensor([[[2.0, 2.0, 2.0]] * 5])

    # Use side_effect to return a different output on each call to the model
    mock_model.side_effect = [mock_output_1, mock_output_2]
    
    sequences = ["SEQ1", "SEQ2"]
    
    # Run with batch_size=1 to ensure it loops twice
    embeddings = generate_embeddings(sequences, mock_tokenizer, mock_model, batch_size=1)
    
    assert len(embeddings) == 2
    assert embeddings[0] == [1.0, 1.0, 1.0]
    assert embeddings[1] == [2.0, 2.0, 2.0]
    
    # Verify tokenizer and model were each called twice
    assert mock_tokenizer.call_count == 2
    assert mock_model.call_count == 2


def test_protein_embeddings_asset(mocker):
    """Tests the asset orchestration by mocking helper functions."""
    # Mock helpers
    mock_load_seqs = mocker.patch("open_target_graph.assets.modeling.inference.load_sequences")
    mock_load_seqs.return_value = (["ID1"], ["SEQ1"])
    
    mock_load_model = mocker.patch("open_target_graph.assets.modeling.inference.load_model")
    mock_load_model.return_value = (MagicMock(), MagicMock())
    
    mock_gen_embeddings = mocker.patch("open_target_graph.assets.modeling.inference.generate_embeddings")
    mock_gen_embeddings.return_value = [[0.1, 0.2]]
    
    mock_save = mocker.patch("open_target_graph.assets.modeling.inference.save_embeddings")
    
    # Run asset
    context = build_asset_context()
    result = protein_embeddings(context, uniprot_parquet="dummy.parquet")
    
    assert result == "data/embeddings.parquet"
    
    # Verify calls
    mock_load_seqs.assert_called_once_with("dummy.parquet")
    mock_load_model.assert_called_once_with(MODEL_NAME)
    mock_gen_embeddings.assert_called_once()
    mock_save.assert_called_once_with(["ID1"], [[0.1, 0.2]], "data/embeddings.parquet")
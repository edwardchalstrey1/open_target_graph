import polars as pl
import torch
from transformers import AutoTokenizer, AutoModel
from dagster import asset, AssetExecutionContext
from typing import List, Tuple, Any

# We use the smallest ESM-2 model for this demo so it runs fast locally.
# In production, you would swap this for "facebook/esm2_t33_650M_UR50D"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

def load_sequences(parquet_path: str) -> Tuple[List[str], List[str]]:
    """Loads sequences and IDs from the input parquet file."""
    df = pl.read_parquet(parquet_path)
    return df["uniprot_id"].to_list(), df["sequence"].to_list()


def load_model(model_name: str) -> Tuple[Any, Any]:
    """Loads the tokenizer and model from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def generate_embeddings(
    sequences: List[str], 
    tokenizer: Any, 
    model: Any, 
    batch_size: int = 4, 
    logger: Any = None
) -> List[List[float]]:
    """Runs inference on a list of sequences to generate embeddings."""
    embeddings = []
    
    if logger:
        logger.info(f"Starting inference on {len(sequences)} sequences...")
    
    with torch.no_grad(): # Disable gradient calculation to save memory
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_seqs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Forward pass (Run the model)
            outputs = model(**inputs)
            
            # Get the "Last Hidden State" (Dimensions: Batch x Seq_Length x Vector_Size)
            # We take the mean across the sequence length to get one vector per protein
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Convert to standard Python lists
            embeddings.extend(batch_embeddings.tolist())
            
            if logger and i % 20 == 0:
                logger.info(f"Processed {i}/{len(sequences)}...")
    
    return embeddings


def save_embeddings(ids: List[str], embeddings: List[List[float]], output_path: str) -> None:
    """Saves the generated embeddings to a parquet file."""
    embedding_df = pl.DataFrame({
        "uniprot_id": ids,
        "embedding": embeddings
    })
    embedding_df.write_parquet(output_path)


@asset(
    group_name="modeling",
    description="Generates vector embeddings for protein sequences using ESM-2"
)
def protein_embeddings(context: AssetExecutionContext, uniprot_parquet: str) -> str:
    """
    This asset takes the raw kinase sequences, runs them through the ESM-2 model to generate embeddings, and saves the results.
    Args:
        context: The Dagster AssetExecutionContext.
        uniprot_parquet: The path to the parquet file containing the raw kinase sequences.
    Returns:
        The path to the saved parquet file with embeddings.
    """
    # 1. Load Data
    ids, sequences = load_sequences(uniprot_parquet)
    
    context.log.info(f"Loading model: {MODEL_NAME}...")
    tokenizer, model = load_model(MODEL_NAME)
    
    # 2. Generate Embeddings
    embeddings = generate_embeddings(sequences, tokenizer, model, batch_size=4, logger=context.log)
    
    # 3. Save Results
    save_path = "data/embeddings.parquet"
    save_embeddings(ids, embeddings, save_path)
    
    context.log.info(f"Saved {len(ids)} embeddings to {save_path}")
    return save_path
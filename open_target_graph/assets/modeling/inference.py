import polars as pl
import torch
from transformers import AutoTokenizer, AutoModel
from dagster import asset, AssetExecutionContext

# We use the smallest ESM-2 model for this demo so it runs fast locally.
# In production, you would swap this for "facebook/esm2_t33_650M_UR50D"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

@asset(
    group_name="modeling",
    description="Generates vector embeddings for protein sequences using ESM-2"
)
def protein_embeddings(context: AssetExecutionContext, uniprot_parquet: str) -> str:
    # 1. Load the raw data
    df = pl.read_parquet(uniprot_parquet)
    sequences = df["sequence"].to_list()
    ids = df["uniprot_id"].to_list()
    
    context.log.info(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    # 2. Process in batches to avoid running out of RAM
    batch_size = 4
    embeddings = []
    
    context.log.info(f"Starting inference on {len(sequences)} sequences...")
    
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
            
            if i % 20 == 0:
                context.log.info(f"Processed {i}/{len(sequences)}...")

    # 3. Save results
    # We combine the ID with the Vector so we can look it up later
    embedding_df = pl.DataFrame({
        "uniprot_id": ids,
        "embedding": embeddings
    })
    
    save_path = "data/embeddings.parquet"
    embedding_df.write_parquet(save_path)
    
    context.log.info(f"Saved {len(embedding_df)} embeddings to {save_path}")
    return save_path
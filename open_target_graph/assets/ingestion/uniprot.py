import polars as pl
import requests
from dagster import asset, AssetExecutionContext, Config

class UniprotConfig(Config):
    num_kinases: int = 100

@asset(
    group_name="ingestion",
    description="Fetches Human Kinase proteins from UniProt API and converts to Polars DataFrame"
)
def raw_uniprot_kinases(context: AssetExecutionContext, config: UniprotConfig) -> pl.DataFrame:
    """
    This asset fetches human kinase protein data from the UniProt API, parses the relevant fields, and returns a Polars DataFrame.
    Args:
        context: The Dagster AssetExecutionContext for logging and asset management.
        config: Configuration for the asset, including the number of kinases to fetch.
    Returns:
        pl.DataFrame: A DataFrame containing UniProt IDs, protein names, gene names, sequences, and sequence lengths for human kinase proteins.
    """
    # UniProt API Query: Human (9606) AND Family:Kinase
    url = f"https://rest.uniprot.org/uniprotkb/search?query=(taxonomy_id:9606)%20AND%20(family:kinase)&format=json&size={config.num_kinases}"
    
    context.log.info(f"Fetching data from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    
    results = response.json()["results"]
    
    # Parse relevant fields into a list of dicts
    parsed_data = []
    for entry in results:
        parsed_data.append({
            "uniprot_id": entry["primaryAccession"],
            "protein_name": entry["proteinDescription"]["recommendedName"]["fullName"]["value"],
            "gene_name": entry["genes"][0]["geneName"]["value"] if entry.get("genes") else None,
            "sequence": entry["sequence"]["value"],
            "length": entry["sequence"]["length"]
        })
        
    # Convert to Polars DataFrame
    df = pl.DataFrame(parsed_data)
    
    context.log.info(f"Ingested {len(df)} kinase targets.")
    return df

@asset(
    group_name="ingestion",
    description="Saves the raw UniProt data to Parquet for downstream processing"
)
def uniprot_parquet(context: AssetExecutionContext, raw_uniprot_kinases: pl.DataFrame):
    """
    This asset saves the raw UniProt data to a Parquet file for downstream processing.
    Args:
        context: The Dagster AssetExecutionContext for logging and asset management.
        raw_uniprot_kinases: The Polars DataFrame containing the raw UniProt kinase data.
    Returns:
        str: The path to the saved Parquet file.
    """
    save_path = "data/kinases.parquet"
    raw_uniprot_kinases.write_parquet(save_path)
    context.log.info(f"Saved parquet file to {save_path}")
    return save_path
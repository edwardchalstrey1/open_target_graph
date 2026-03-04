import polars as pl
from dagster import asset, AssetIn, AssetExecutionContext
from sqlalchemy import create_engine, text

import os

def get_db_connection_string() -> str:
    user = os.environ.get("DB_USER", "admin")
    password = os.environ.get("DB_PASSWORD", "password")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME", "open_target_graph")
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

@asset(
    group_name="db",
    description="Loads parquet data into Postgres with vector extension enabled.",
    ins={
        "uniprot_parquet": AssetIn(key="uniprot_parquet"),
        "protein_embeddings": AssetIn(key="protein_embeddings"),
        "chembl_activity_parquet": AssetIn(key="chembl_activity_parquet")
    }
)
def load_to_postgres(context: AssetExecutionContext, uniprot_parquet: str, protein_embeddings: str, chembl_activity_parquet: str) -> None:
    engine = create_engine(get_db_connection_string())
    
    with engine.begin() as conn:
        # Enable pgvector
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # 1. Load Uniprot Kinases
        context.log.info(f"Loading kinases from {uniprot_parquet}...")
        kinases_df = pl.read_parquet(uniprot_parquet).to_pandas()
        kinases_df.to_sql("kinases", conn, if_exists="replace", index=False)
        
        # 2. Load Embeddings with pgvector
        context.log.info(f"Loading embeddings from {protein_embeddings}...")
        emb_df = pl.read_parquet(protein_embeddings)
        
        # Create vectors table manually because Pandas to_sql doesn't support vector types natively
        conn.execute(text("DROP TABLE IF EXISTS embeddings;"))
        conn.execute(text("""
            CREATE TABLE embeddings (
                uniprot_id TEXT PRIMARY KEY,
                embedding vector(320)
            );
        """))
        
        insert_query = text("INSERT INTO embeddings (uniprot_id, embedding) VALUES (:uniprot_id, :embedding)")
        for row in emb_df.iter_rows(named=True):
            conn.execute(insert_query, {
                "uniprot_id": row["uniprot_id"],
                "embedding": row["embedding"]
            })
            
        # 3. Load ChEMBL Activity
        if os.path.exists(chembl_activity_parquet):
            context.log.info(f"Loading activities from {chembl_activity_parquet}...")
            chembl_df = pl.read_parquet(chembl_activity_parquet).to_pandas()
            if not chembl_df.empty:
                chembl_df.to_sql("chembl_activities", conn, if_exists="replace", index=False)
        
    context.log.info("Finished loading data to Postgres.")

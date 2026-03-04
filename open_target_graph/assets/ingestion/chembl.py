import polars as pl
import requests
from dagster import asset, AssetIn, AssetExecutionContext
from typing import List, Dict, Any, Optional
import os

CHEMBL_API_URL = "https://www.ebi.ac.uk/chembl/api/data"

def get_target_chembl_id(accession: str, logger: Any = None) -> Optional[str]:
    """
    Fetches the target ChEMBL ID for a given UniProt accession.
    This is more robust than using a deep filter on the activity endpoint.
    """
    url = f"{CHEMBL_API_URL}/target/search.json?q={accession}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("targets"):
            # The first result is usually the correct one for a specific accession
            return data["targets"][0].get("target_chembl_id")
    except requests.exceptions.RequestException as e:
        if logger:
            logger.warning(f"Could not fetch ChEMBL ID for accession {accession}. Error: {e}. Skipping.")
    return None

def fetch_activities_for_chembl_id(chembl_id: str) -> List[Dict[str, Any]]:
    """Fetches all activities for a given ChEMBL ID with pagination."""
    # We filter for a pChEMBL value to get quantitative data
    # and for assays with high confidence to ensure data quality.
    url = (
        f"{CHEMBL_API_URL}/activity.json?target_chembl_id={chembl_id}"
        "&pchembl_value__isnull=false&assay_confidence_score__gte=5"
    )
    
    activities = []
    while url:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        activities.extend(data["activities"])
        # Follow the pagination link if it exists
        if data["page_meta"]["next"]:
            # The 'next' URL is relative, so we need to prepend the base
            url = f"https://www.ebi.ac.uk{data['page_meta']['next']}"
        else:
            url = None
    return activities

def fetch_molecule_details(molecule_ids: List[str], logger: Any = None) -> Dict[str, Dict[str, Any]]:
    """Fetches details for a list of molecule ChEMBL IDs in batches."""
    batch_size = 50  # ChEMBL API is limited in how many IDs can be passed at once.
    molecule_details = {}
    
    for i in range(0, len(molecule_ids), batch_size):
        batch_ids = molecule_ids[i:i+batch_size]
        ids_str = ",".join(batch_ids)
        url = f"{CHEMBL_API_URL}/molecule.json?molecule_chembl_id__in={ids_str}&only=molecule_chembl_id,pref_name,molecule_structures"
        
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            if logger:
                logger.warning(f"A batch of ChEMBL molecule details failed to fetch. Error: {e}. Skipping batch.")
            continue
        
        for mol in data.get("molecules", []):
            molecule_details[mol["molecule_chembl_id"]] = {
                "pref_name": mol.get("pref_name"),
                "canonical_smiles": (mol.get("molecule_structures") or {}).get("canonical_smiles")
            }
    return molecule_details

@asset(
    group_name="ingestion",
    description="Fetches bioactive molecules from ChEMBL for known kinase targets.",
    ins={"uniprot_parquet": AssetIn(key="uniprot_parquet")}
)
def chembl_activity_parquet(context: AssetExecutionContext, uniprot_parquet: str) -> str:
    """
    For each kinase, fetches associated bioactivity data from ChEMBL,
    enriches it with molecule details, and saves it as a Parquet file.
    Resumes from previous runs if they failed.
    """
    context.log.info(f"Loading kinase IDs from {uniprot_parquet}...")
    uniprot_df = pl.read_parquet(uniprot_parquet)
    kinase_ids = uniprot_df["uniprot_id"].to_list()
    
    save_path = "data/chembl_activity.parquet"
    processed_path = "data/chembl_processed_kinases.txt"
    
    processed_ids = set()
    existing_df = None
    
    if os.path.exists(save_path):
        existing_df = pl.read_parquet(save_path)
        if "uniprot_id" in existing_df.columns:
            processed_ids.update(existing_df["uniprot_id"].unique().to_list())
            
    if os.path.exists(processed_path):
        with open(processed_path, "r") as f:
            processed_ids.update(line.strip() for line in f if line.strip())

    all_activities = []
    
    kinases_to_process = [k for k in kinase_ids if k not in processed_ids]
    context.log.info(f"Fetching ChEMBL activities for {len(kinases_to_process)} kinases ({len(processed_ids)} already processed)...")
    
    for i, uniprot_id in enumerate(kinases_to_process):
        target_chembl_id = get_target_chembl_id(uniprot_id, logger=context.log)
        
        # Mark as processed whether we found it or not
        with open(processed_path, "a") as f:
            f.write(f"{uniprot_id}\n")

        if not target_chembl_id:
            continue # Skip if we couldn't find a mapping to a ChEMBL ID

        try:
            activities = fetch_activities_for_chembl_id(target_chembl_id)
            for act in activities:
                all_activities.append({
                    "uniprot_id": uniprot_id,
                    "molecule_chembl_id": act.get("molecule_chembl_id"),
                    "pchembl_value": float(act.get("pchembl_value")) if act.get("pchembl_value") else None,
                    "standard_type": act.get("standard_type"),
                })
        except requests.exceptions.RequestException as e:
            context.log.warning(f"Failed to fetch activities for ChEMBL ID {target_chembl_id} (from UniProt {uniprot_id}). Error: {e}. Skipping.")
            continue # Move to the next kinase
            
        if (i + 1) % 10 == 0:
            context.log.info(f"Processed {i+1}/{len(kinases_to_process)} kinases...")

    if not all_activities:
        if existing_df is not None and not existing_df.is_empty():
            context.log.info("No new activities found. Keeping existing DataFrame.")
            return save_path
        else:
            context.log.warning("No activities found. Saving empty DataFrame.")
            pl.DataFrame({}).write_parquet(save_path)
            return save_path

    new_activity_df = pl.DataFrame(all_activities).drop_nulls("molecule_chembl_id")
    
    unique_molecule_ids = new_activity_df["molecule_chembl_id"].unique().to_list()
    context.log.info(f"Fetching details for {len(unique_molecule_ids)} unique molecules...")
    molecule_map = fetch_molecule_details(unique_molecule_ids, logger=context.log)
    
    mol_df = pl.DataFrame([{"molecule_chembl_id": k, **v} for k, v in molecule_map.items()])
    new_final_df = new_activity_df.join(mol_df, on="molecule_chembl_id")
    
    if existing_df is not None and not existing_df.is_empty():
        # Ensure schemas match before concat, just in case
        for col in new_final_df.columns:
            if col not in existing_df.columns:
                existing_df = existing_df.with_columns(pl.lit(None).alias(col))
        for col in existing_df.columns:
            if col not in new_final_df.columns:
                new_final_df = new_final_df.with_columns(pl.lit(None).alias(col))
        
        final_df = pl.concat([existing_df.select(new_final_df.columns), new_final_df])
    else:
        final_df = new_final_df
        
    final_df.write_parquet(save_path)
    context.log.info(f"Saved {len(final_df)} activity records to {save_path}")
    return save_path
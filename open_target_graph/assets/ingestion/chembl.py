import polars as pl
import requests
from dagster import asset, AssetIn, AssetExecutionContext
from typing import List, Dict, Any

CHEMBL_API_URL = "https://www.ebi.ac.uk/chembl/api/data"

def fetch_activities_for_accession(accession: str) -> List[Dict[str, Any]]:
    """Fetches all activities for a given UniProt accession with pagination."""
    # We filter for a pChEMBL value to get quantitative data
    # and for assays with high confidence to ensure data quality.
    url = (
        f"{CHEMBL_API_URL}/activity.json?target_chembl_id__target_components__accession={accession}"
        "&pchembl_value__isnull=false&assay_confidence_score__gte=5"
    )
    
    activities = []
    while url:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        activities.extend(data["activities"])
        # Follow the pagination link if it exists
        if data["page_meta"]["next"]:
            url = f"https://www.ebi.ac.uk{data['page_meta']['next']}"
        else:
            url = None
    return activities

def fetch_molecule_details(molecule_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetches details for a list of molecule ChEMBL IDs in batches."""
    batch_size = 50  # ChEMBL API is limited in how many IDs can be passed at once.
    molecule_details = {}
    
    for i in range(0, len(molecule_ids), batch_size):
        batch_ids = molecule_ids[i:i+batch_size]
        ids_str = ",".join(batch_ids)
        url = f"{CHEMBL_API_URL}/molecule.json?molecule_chembl_id__in={ids_str}&only=molecule_chembl_id,pref_name,molecule_structures"
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        for mol in data.get("molecules", []):
            molecule_details[mol["molecule_chembl_id"]] = {
                "pref_name": mol.get("pref_name"),
                "canonical_smiles": mol.get("molecule_structures", {}).get("canonical_smiles")
            }
    return molecule_details

@asset(
    group_name="ingestion",
    description="Fetches bioactive molecules from ChEMBL for known kinase targets.",
    ins={"uniprot_parquet": AssetIn(key="uniprot_parquet")}
)
def chembl_activity_parquet(context: AssetExecutionContext, uniprot_parquet: pl.DataFrame) -> str:
    """
    For each kinase, fetches associated bioactivity data from ChEMBL,
    enriches it with molecule details, and saves it as a Parquet file.
    """
    kinase_ids = uniprot_parquet["uniprot_id"].to_list()
    all_activities = []
    
    context.log.info(f"Fetching ChEMBL activities for {len(kinase_ids)} kinases...")
    
    for i, uniprot_id in enumerate(kinase_ids):
        activities = fetch_activities_for_accession(uniprot_id)
        for act in activities:
            all_activities.append({
                "uniprot_id": uniprot_id,
                "molecule_chembl_id": act.get("molecule_chembl_id"),
                "pchembl_value": float(act.get("pchembl_value")) if act.get("pchembl_value") else None,
                "standard_type": act.get("standard_type"),
            })
        if (i + 1) % 10 == 0:
            context.log.info(f"Processed {i+1}/{len(kinase_ids)} kinases...")

    if not all_activities:
        context.log.warning("No activities found. Saving empty DataFrame.")
        save_path = "data/chembl_activity.parquet"
        pl.DataFrame({}).write_parquet(save_path)
        return save_path

    activity_df = pl.DataFrame(all_activities).drop_nulls("molecule_chembl_id")
    
    unique_molecule_ids = activity_df["molecule_chembl_id"].unique().to_list()
    context.log.info(f"Fetching details for {len(unique_molecule_ids)} unique molecules...")
    molecule_map = fetch_molecule_details(unique_molecule_ids)
    
    mol_df = pl.DataFrame([{"molecule_chembl_id": k, **v} for k, v in molecule_map.items()])
    final_df = activity_df.join(mol_df, on="molecule_chembl_id")
    
    save_path = "data/chembl_activity.parquet"
    final_df.write_parquet(save_path)
    context.log.info(f"Saved {len(final_df)} activity records to {save_path}")
    return save_path
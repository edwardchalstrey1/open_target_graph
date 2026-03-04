import polars as pl
import requests
from dagster import asset, AssetIn, AssetExecutionContext
from typing import List, Dict, Any, Optional

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
        except requests.exceptions.HTTPError as e:
            if logger:
                logger.warning(f"A batch of ChEMBL molecule details failed to fetch. Status: {e.response.status_code}. Skipping batch.")
            continue
        
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
def chembl_activity_parquet(context: AssetExecutionContext, uniprot_parquet: str) -> str:
    """
    For each kinase, fetches associated bioactivity data from ChEMBL,
    enriches it with molecule details, and saves it as a Parquet file.
    """
    context.log.info(f"Loading kinase IDs from {uniprot_parquet}...")
    uniprot_df = pl.read_parquet(uniprot_parquet)
    kinase_ids = uniprot_df["uniprot_id"].to_list()
    all_activities = []
    
    context.log.info(f"Fetching ChEMBL activities for {len(kinase_ids)} kinases...")
    
    for i, uniprot_id in enumerate(kinase_ids):
        target_chembl_id = get_target_chembl_id(uniprot_id, logger=context.log)
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
        except requests.exceptions.HTTPError as e:
            context.log.warning(f"Failed to fetch activities for ChEMBL ID {target_chembl_id} (from UniProt {uniprot_id}). Status: {e.response.status_code}. Skipping.")
            continue # Move to the next kinase
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
    molecule_map = fetch_molecule_details(unique_molecule_ids, logger=context.log)
    
    mol_df = pl.DataFrame([{"molecule_chembl_id": k, **v} for k, v in molecule_map.items()])
    final_df = activity_df.join(mol_df, on="molecule_chembl_id")
    
    save_path = "data/chembl_activity.parquet"
    final_df.write_parquet(save_path)
    context.log.info(f"Saved {len(final_df)} activity records to {save_path}")
    return save_path
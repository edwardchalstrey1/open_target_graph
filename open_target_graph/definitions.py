from dagster import Definitions, load_assets_from_modules
from open_target_graph.assets.ingestion import uniprot

# Load assets specifically from that file
ingestion_assets = load_assets_from_modules([uniprot])

defs = Definitions(
    assets=[*ingestion_assets],
)
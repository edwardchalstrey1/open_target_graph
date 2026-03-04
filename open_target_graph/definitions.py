from dagster import Definitions, load_assets_from_modules
from open_target_graph.assets.ingestion import uniprot, chembl
from open_target_graph.assets.modeling import inference
from open_target_graph.assets.db import postgres


ingestion_assets = load_assets_from_modules([uniprot, chembl])
modeling_assets = load_assets_from_modules([inference])
db_assets = load_assets_from_modules([postgres])

defs = Definitions(
    # Combine the lists using the * unpacking operator
    assets=[*ingestion_assets, *modeling_assets, *db_assets],
)
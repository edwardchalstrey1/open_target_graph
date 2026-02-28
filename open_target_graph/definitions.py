from dagster import Definitions, load_assets_from_modules
from open_target_graph.assets.ingestion import uniprot
from open_target_graph.assets.modeling import inference


ingestion_assets = load_assets_from_modules([uniprot])
modeling_assets = load_assets_from_modules([inference])

defs = Definitions(
    # Combine the lists using the * unpacking operator
    assets=[*ingestion_assets, *modeling_assets],
)
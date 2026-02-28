import streamlit as st
import polars as pl
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from stmol import showmol
import py3Dmol
import requests
from typing import Optional, List

# Constants
DATA_PATH_KINASES = "data/kinases.parquet"
DATA_PATH_EMBEDDINGS = "data/embeddings.parquet"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"

def configure_page() -> None:
    """Configures the Streamlit page settings."""
    st.set_page_config(layout="wide", page_title="OpenTargetGraph")

def render_header() -> None:
    """Renders the dashboard header and description."""
    st.title("🧬 OpenTargetGraph: AI-Driven Target Discovery")
    st.markdown("""
    This dashboard visualizes **Kinase targets** and their structural similarity using **ESM-2 Embeddings**. 
    Instead of traditional sequence alignment, we use a **Protein Language Model** to capture deep semantic relationships between proteins.

    #### 🤖 Model: ESM-2
    **ESM-2** is a transformer-based model trained on millions of protein sequences. It converts a protein sequence into an embedding that encodes structural and functional properties.
    """)

# --- Load Data ---
@st.cache_data
def load_data() -> pl.DataFrame:
    """
    Loads kinase metadata and embeddings from Parquet files.

    Returns:
        pl.DataFrame: Joined DataFrame containing metadata and embeddings.
    """
    try:
        df_meta = pl.read_parquet(DATA_PATH_KINASES)
        df_emb = pl.read_parquet(DATA_PATH_EMBEDDINGS)
        
        # Join them on ID
        return df_meta.join(df_emb, on="uniprot_id")
    except Exception as e:
        st.error(f"Data not found or error loading data: {e}")
        st.stop()

def fetch_pdb_data(uniprot_id: str) -> Optional[str]:
    """
    Fetches PDB structure data from AlphaFold DB for a given UniProt ID.

    Args:
        uniprot_id (str): The UniProt accession ID.

    Returns:
        Optional[str]: The PDB file content as a string, or None if not found.
    """
    api_url = ALPHAFOLD_API_URL.format(uniprot_id)
    try:
        api_response = requests.get(api_url, timeout=5)
        if api_response.ok and len(api_response.json()) > 0:
            pdb_url = api_response.json()[0]["pdbUrl"]
            pdb_response = requests.get(pdb_url, timeout=10)
            if pdb_response.ok:
                return pdb_response.text
    except requests.RequestException:
        return None
    return None

def create_3d_view(pdb_data: str, width: int = 400, height: int = 300) -> py3Dmol.view:
    """
    Creates a py3Dmol view object for a given PDB string.

    Args:
        pdb_data (str): The PDB file content.
        width (int): Width of the viewer.
        height (int): Height of the viewer.

    Returns:
        py3Dmol.view: The configured 3D view object.
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, "pdb")
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    return view

def render_target_selection(df: pl.DataFrame) -> str:
    """
    Renders the target selection dropdown and details.

    Args:
        df (pl.DataFrame): The main dataframe.

    Returns:
        str: The selected UniProt ID.
    """
    st.subheader("1. Select a Target")
    
    options = df["uniprot_id"].to_list()
    
    def format_func(uid: str) -> str:
        # Optimization: In a real app, create a dict for O(1) lookup instead of filtering df every time
        name = df.filter(pl.col('uniprot_id') == uid)['protein_name'][0]
        return f"{uid} - {name}"

    selected_id = st.selectbox(
        "Choose a protein:", 
        options,
        format_func=format_func
    )
    
    # Get details
    target_row = df.filter(pl.col("uniprot_id") == selected_id)
    seq = target_row["sequence"][0]
    
    st.text_area("Sequence", seq, height=100)
    return selected_id

def render_structure_preview(selected_id: str) -> None:
    """
    Renders the 3D structure preview section.

    Args:
        selected_id (str): The UniProt ID to visualize.
    """
    st.subheader("3D Structure Preview")
    
    pdb_data = fetch_pdb_data(selected_id)
    
    if pdb_data:
        view = create_3d_view(pdb_data)
        showmol(view, height=300, width=400)
    else:
        st.warning(f"Structure not found for {selected_id}")
        # Show empty viewer to maintain layout
        view = py3Dmol.view(width=400, height=300)
        showmol(view, height=300, width=400)

def compute_tsne_projection(embeddings: List[List[float]], perplexity: int = 30) -> np.ndarray:
    """
    Computes t-SNE projections for high-dimensional embeddings.

    Args:
        embeddings (List[List[float]]): List of embedding vectors.
        perplexity (int): t-SNE perplexity parameter.

    Returns:
        np.ndarray: 2D array of projected coordinates.
    """
    matrix = np.array(embeddings)
    safe_perplexity = min(perplexity, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=safe_perplexity)
    return tsne.fit_transform(matrix)

def render_tsne_plot(df: pl.DataFrame, selected_id: str) -> None:
    """
    Renders the t-SNE embedding space visualization.

    Args:
        df (pl.DataFrame): The main dataframe containing embeddings.
        selected_id (str): The currently selected UniProt ID to highlight.
    """
    st.subheader("2. Embedding Space (t-SNE)")
    st.markdown("""
    **What is this plot?**
    We use **t-SNE** (t-Distributed Stochastic Neighbor Embedding) to project the 320-dimensional ESM-2 vectors down to 2D.
    * **Points**: Each dot is a Kinase protein.
    * **Proximity**: Points closer together are "semantically" similar in the eyes of the AI model.
    """)

    if st.button("Generate Plot"):
        with st.spinner("Projecting 320-dim vectors to 2D..."):
            embeddings = df["embedding"].to_list()
            projections = compute_tsne_projection(embeddings)
            
            # Add to dataframe for plotting
            plot_df = df.with_columns([
                pl.Series("x", projections[:, 0]),
                pl.Series("y", projections[:, 1])
            ])
            
            # Plot
            fig = px.scatter(
                plot_df.to_pandas(), 
                x="x", y="y", 
                labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2'},
                hover_data=["uniprot_id", "protein_name", "gene_name"],
                color="length", 
                title="Protein Similarity Map (ESM-2 Latent Space)"
            )
            
            # Highlight selected point
            selected_point = plot_df.filter(pl.col("uniprot_id") == selected_id)
            if not selected_point.is_empty():
                fig.add_scatter(
                    x=selected_point["x"], 
                    y=selected_point["y"], 
                    mode='markers', 
                    marker=dict(size=15, color='red', symbol='x'),
                    name='Selected'
                )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click the button to run t-SNE projection.")

def main() -> None:
    """Main execution entry point."""
    configure_page()
    render_header()
    
    df = load_data()
    st.success(f"Loaded {len(df)} targets with embeddings.")
    
    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        selected_id = render_target_selection(df)
        
    with col2:
        render_structure_preview(selected_id)
    
    st.divider()
    render_tsne_plot(df, selected_id)

if __name__ == "__main__":
    main()

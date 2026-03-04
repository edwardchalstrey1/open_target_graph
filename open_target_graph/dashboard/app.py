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
DATA_PATH_CHEMBL = "data/chembl_activity.parquet"
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

    #### 🤖 Model: [ESM-2](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
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

@st.cache_data
def load_chembl_data() -> pl.DataFrame:
    """Loads ChEMBL activity data from a Parquet file."""
    try:
        df = pl.read_parquet(DATA_PATH_CHEMBL)
        if df.is_empty():
            st.warning("ChEMBL data is empty. Drug candidate search will be unavailable.")
        return df
    except Exception:
        st.warning("ChEMBL data not found. Please run the `chembl_activity_parquet` asset. Drug candidate search will be unavailable.")
        return pl.DataFrame()

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

def find_similar_targets(df: pl.DataFrame, selected_id: str, top_n: int = 5) -> pl.DataFrame:
    """
    Finds the most similar targets to a selected protein based on cosine similarity of their embeddings.

    Args:
        df (pl.DataFrame): The DataFrame with embeddings.
        selected_id (str): The UniProt ID of the protein to compare against.
        top_n (int): The number of similar targets to return.

    Returns:
        pl.DataFrame: A DataFrame of the top N similar targets and their similarity scores.
    """
    # 1. Get the embedding for the selected target.
    target_embedding = df.filter(pl.col("uniprot_id") == selected_id)["embedding"][0]

    # 2. Calculate cosine similarity using pure Polars expressions for performance.
    # Cosine Similarity = (A · B) / (||A|| * ||B||)
    
    # A · B (dot product)
    dot_product = pl.col("embedding").list.eval(
        pl.element() * pl.lit(pl.Series(target_embedding))
    ).list.sum()

    # ||A|| (L2 norm of target)
    norm_target = np.linalg.norm(np.array(target_embedding))

    # ||B|| (L2 norm of other vectors in the column)
    norm_other = pl.col("embedding").list.eval(
        pl.element().pow(2)
    ).list.sum().sqrt()

    similarity_expr = (dot_product / (pl.lit(norm_target) * norm_other)).alias("similarity")

    # 3. Apply expression, sort, filter out the original protein, and take the top N.
    return df.with_columns(
        similarity_expr
    ).sort("similarity", descending=True).filter(
        pl.col("uniprot_id") != selected_id
    ).head(top_n)

def render_target_selection(df: pl.DataFrame) -> None:
    """
    Renders the target selection dropdown and details.

    Args:
        df (pl.DataFrame): The main dataframe.
    """
    st.subheader("Select a Target")
    
    options = df["uniprot_id"].to_list()
    
    def format_func(uid: str) -> str:
        # Optimization: In a real app, create a dict for O(1) lookup instead of filtering df every time
        name = df.filter(pl.col('uniprot_id') == uid)['protein_name'][0]
        return f"{uid} - {name}"

    # Use a key to link this widget to the session state
    st.selectbox(
        "Choose a protein:", 
        options,
        format_func=format_func,
        key="selected_id"
    )
    
    # Get details
    target_row = df.filter(pl.col("uniprot_id") == st.session_state.selected_id)
    seq = target_row["sequence"][0]
    
    st.text_area("Sequence", seq, height=100)

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

def render_similarity_search(df: pl.DataFrame, selected_id: str) -> pl.DataFrame:
    """
    Renders the UI for finding and displaying similar targets.

    Args:
        df (pl.DataFrame): The main dataframe.
        selected_id (str): The currently selected UniProt ID.

    Returns:
        pl.DataFrame: The dataframe containing top similar targets.
    """
    st.subheader("3. Find Similar Targets & Drug Candidates")
    st.markdown("Find the most similar proteins in the high-dimensional embedding space using cosine similarity. This is more powerful than sequence alignment as it captures functional and structural relationships learned by the ESM-2 model.")

    similar_targets = find_similar_targets(df, selected_id, top_n=5)
    
    st.write("Most similar targets:")
    st.dataframe(
        similar_targets.select("uniprot_id", "protein_name", "gene_name", "similarity")
        .to_pandas()
        .style.format({"similarity": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    return similar_targets

def render_drug_candidates(chembl_df: pl.DataFrame, similar_targets_df: pl.DataFrame, selected_id: str) -> None:
    """Renders a table of drug candidates for the selected target and its neighbors."""
    st.markdown("#### Semantic Drug Candidate Search")
    st.markdown(
        """
    Based on protein embedding similarity, we can infer that drugs targeting similar proteins might also be effective. 
    This table shows known bioactive molecules for your selected target and its closest neighbors in the embedding space.
    """
    )

    if chembl_df.is_empty():
        return

    neighbor_ids = similar_targets_df["uniprot_id"].to_list()
    all_relevant_ids = [selected_id] + neighbor_ids

    candidate_drugs = chembl_df.filter(pl.col("uniprot_id").is_in(all_relevant_ids)).sort("pchembl_value", descending=True)

    if candidate_drugs.is_empty():
        st.info("No known bioactive molecules found in ChEMBL for the selected target or its neighbors.")
        return

    candidate_drugs = candidate_drugs.with_columns(
        pl.when(pl.col("uniprot_id") == selected_id).then(pl.lit("Direct Target")).otherwise(pl.lit("Inferred from Neighbor")).alias("Candidate Type")
    ).with_columns(
        pl.when(pl.col("Candidate Type") == "Inferred from Neighbor").then(pl.col("uniprot_id")).otherwise(None).alias("Source Target")
    )

    display_df = candidate_drugs.select(
        ["Candidate Type", "pref_name", "pchembl_value", "standard_type", "Source Target"]
    ).rename({"pref_name": "Molecule Name", "pchembl_value": "pChEMBL"})

    st.dataframe(display_df, use_container_width=True, hide_index=True)

@st.cache_data
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

def render_tsne_plot(df: pl.DataFrame) -> None:
    """
    Renders the t-SNE embedding space visualization.

    Args:
        df (pl.DataFrame): The main dataframe containing embeddings.
    """
    st.subheader("Embedding Space (t-SNE)")
    st.markdown("""
    **What is this plot?**
    We use **t-SNE** (t-Distributed Stochastic Neighbor Embedding) to project the 320-dimensional ESM-2 vectors down to 2D.
    * **Points**: Each dot is a Kinase protein.
    * **Proximity**: Points closer together are "semantically" similar in the eyes of the AI model.
    """)

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
        hover_data=["protein_name", "gene_name"],
        custom_data=["uniprot_id"],
        color="length", 
        title="Protein Similarity Map (ESM-2 Latent Space)"
    )
    
    # Highlight selected point
    selected_point = plot_df.filter(pl.col("uniprot_id") == st.session_state.selected_id)
    if not selected_point.is_empty():
        fig.add_scatter(
            x=selected_point["x"], 
            y=selected_point["y"], 
            mode='markers', 
            marker=dict(size=15, color='red', symbol='x'),
            name='Selected'
        )
    
    # Enable click events and capture the output
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

    # If a point is clicked, update the session state and rerun the app
    if event and "selection" in event and event["selection"]["points"]:
        point = event["selection"]["points"][0]
        if "customdata" in point:
            clicked_id = point["customdata"][0]
            if st.session_state.selected_id != clicked_id:
                st.session_state.plot_selection = clicked_id
                st.rerun()

def main() -> None:
    """Main execution entry point."""
    configure_page()
    render_header()
    
    df = load_data()
    chembl_df = load_chembl_data()
    st.success(f"Loaded {len(df)} targets and {len(chembl_df)} activity records.")

    # Check if a selection was made from the plot in the previous run
    if "plot_selection" in st.session_state:
        # Update the primary selection key
        st.session_state.selected_id = st.session_state.plot_selection
        # Clean up the temporary key
        del st.session_state.plot_selection

    # Initialize session state for the selected protein if it doesn't exist.
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = df["uniprot_id"][0]
    
    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        render_target_selection(df)
        
    with col2:
        render_structure_preview(st.session_state.selected_id)
    
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        similar_targets_df = render_similarity_search(df, st.session_state.selected_id)
        st.divider()
        render_drug_candidates(chembl_df, similar_targets_df, st.session_state.selected_id)
    with col4:
        render_tsne_plot(df)

if __name__ == "__main__":
    main()

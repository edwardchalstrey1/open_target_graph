import streamlit as st
import polars as pl
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from stmol import showmol
import py3Dmol
import requests
import os
from sqlalchemy import create_engine, text
from typing import Optional, List, Dict, Any
from open_target_graph.agents.workflow import research_app

# Constants
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"

def get_db_connection_string() -> str:
    user = os.environ.get("DB_USER", "admin")
    password = os.environ.get("DB_PASSWORD", "password")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME", "open_target_graph")
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

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
    Loads kinase metadata and embeddings from Postgres.
    """
    try:
        engine = create_engine(get_db_connection_string())
        
        # We need the vector as a list/array for the t-SNE plot later
        query = """
            SELECT k.*, e.embedding::text as embedding_str
            FROM kinases k
            JOIN embeddings e ON k.uniprot_id = e.uniprot_id
        """
        df = pl.read_database(query, engine)
        
        # Convert the string representation of vector '[0.1, 0.2, ...]' back to a list of floats
        df = df.with_columns(
            pl.col("embedding_str")
            .str.replace(r"^\[", "").str.replace(r"\]$", "")
            .str.replace_all(" ", "")
            .str.split(",")
            .list.eval(pl.element().cast(pl.Float64))
            .alias("embedding")
        ).drop("embedding_str")
        
        return df
    except Exception as e:
        st.error(f"Data not found or error loading data. Ensure `load_to_postgres` asset has been run. Error: {e}")
        st.stop()

@st.cache_data
def load_chembl_data() -> pl.DataFrame:
    """Loads ChEMBL activity data from Postgres."""
    try:
        engine = create_engine(get_db_connection_string())
        df = pl.read_database("SELECT * FROM chembl_activities", engine)
        if df.is_empty():
            st.warning("ChEMBL data is empty. Drug candidate search will be unavailable.")
        return df
    except Exception:
        st.warning("ChEMBL data not found. Please run the `load_to_postgres` asset. Drug candidate search will be unavailable.")
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

def create_3d_view(pdb_data: str, width: str = "100%", height: int = 400) -> py3Dmol.view:
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
    Finds the most similar targets to a selected protein using Postgres pgvector cosine distance.
    """
    engine = create_engine(get_db_connection_string())
    
    query = text(f"""
        SELECT 
            k.uniprot_id, 
            k.protein_name, 
            k.gene_name,
            -- pgvector <=> operator computes cosine distance. 
            -- Cosine similarity is 1 - Cosine distance.
            1 - (e.embedding <=> (SELECT embedding FROM embeddings WHERE uniprot_id = :selected_id)) AS similarity
        FROM kinases k
        JOIN embeddings e ON k.uniprot_id = e.uniprot_id
        WHERE k.uniprot_id != :selected_id
        ORDER BY similarity DESC
        LIMIT :top_n
    """)
    
    with engine.connect() as conn:
        result = pl.read_database(query, conn, execute_options={"parameters": {"selected_id": selected_id, "top_n": top_n}})
        
    return result

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
        view = create_3d_view(pdb_data, width="100%", height=500)
        showmol(view, width=None, height=500)
    else:
        st.warning(f"Structure not found for {selected_id}")
        # Show empty viewer to maintain layout
        view = py3Dmol.view(width="100%", height=500)
        showmol(view, width=None, height=500)

def render_similarity_search(df: pl.DataFrame, selected_id: str) -> pl.DataFrame:
    """
    Renders the UI for finding and displaying similar targets.

    Args:
        df (pl.DataFrame): The main dataframe.
        selected_id (str): The currently selected UniProt ID.

    Returns:
        pl.DataFrame: The dataframe containing top similar targets.
    """
    top_n = 5
    st.markdown(f"The table below shows the top {top_n} most similar protein targets in the high-dimensional embedding space using cosine similarity. This is more powerful than sequence alignment as it captures functional and structural relationships learned by the ESM-2 model.")

    similar_targets = find_similar_targets(df, selected_id, top_n=top_n)
    
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
    st.subheader("Semantic Drug Candidate Search")
    st.markdown(
        """
    Based on protein embedding similarity, we can infer that drugs targeting similar proteins might also be effective. 
    This table shows known bioactive molecules for your selected target and its closest neighbors in the embedding space.

    * **Candidate Type**: Whether the drug was found for the "Direct Target" or "Inferred from Neighbor".
    * **Molecule Name**: The ChEMBL preferred name for the bioactive molecule.
    * **pChEMBL**: A standardized measure of a drug's potency (higher is more potent).
    * **standard_type**: The activity type (e.g., IC50, Ki, Kd) the measurement is based on.
    * **Source Target**: Which neighboring protein the drug originally targets (if inferred).
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

def render_ai_research_assistant(uniprot_id: str, protein_name: str) -> None:
    """Renders the AI Research Assistant section."""
    st.divider()
    st.subheader("🤖 Autonomous Research Assistant")
    st.markdown(f"Generate a deep-dive research report for **{protein_name}** using Pydantic AI and PubMed.")
    
    if "research_report" not in st.session_state:
        st.session_state.research_report = None

    if st.button("🚀 Generate AI Research Report", type="primary"):
        if not os.environ.get("GEMINI_API_KEY"):
            st.error("Please set your `GEMINI_API_KEY` environment variable to use the AI Research Assistant.")
            return
            
        with st.status("Analyzing target and searching PubMed...", expanded=True) as status:
            st.write("Initializing agent workflow...")
            inputs = {
                "uniprot_id": uniprot_id,
                "protein_name": protein_name,
                "query": f"{protein_name} drug discovery",
                "raw_papers": [],
                "final_report": {},
                "error": ""
            }
            
            try:
                result = research_app.invoke(inputs)
                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.session_state.research_report = result["final_report"]
                    status.update(label="Report Generated!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Workflow execution failed: {e}")

    if st.session_state.research_report:
        report = st.session_state.research_report
        
        # Display Report
        st.info(f"Report Generated at: {report.get('generated_at')}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Mechanism Summary")
            st.write(report.get("mechanism_summary"))
            
            st.markdown("### Key Scientific Papers")
            for paper in report.get("top_papers", []):
                with st.expander(f"📄 {paper.get('title')} ({paper.get('year')})"):
                    st.write(f"**Authors:** {', '.join(paper.get('authors', []))}")
                    st.write(f"**Key Findings:**")
                    for finding in paper.get("key_findings", []):
                        st.write(f"- {finding}")
                    st.markdown(f"[View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{paper.get('pubmed_id')}/)")
                    st.metric("Relevance", f"{paper.get('relevance_score')}/10")
        
        with col2:
            st.markdown("### Recommendation")
            rec = report.get("recommendation", "N/A")
            if "Go" in rec and "No-Go" not in rec:
                st.success(rec)
            elif "No-Go" in rec:
                st.error(rec)
            else:
                st.warning(rec)
                
            st.markdown("### Clinical Status")
            st.info(report.get("clinical_trial_status", "Unknown"))

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
        similar_targets_df = render_similarity_search(df, st.session_state.selected_id)
        
    with col2:
        render_structure_preview(st.session_state.selected_id)
    
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        render_tsne_plot(df) 
    with col4:
        render_drug_candidates(chembl_df, similar_targets_df, st.session_state.selected_id)

    # Render AI Assistant at the bottom
    target_name = df.filter(pl.col('uniprot_id') == st.session_state.selected_id)['protein_name'][0]
    render_ai_research_assistant(st.session_state.selected_id, target_name)

if __name__ == "__main__":
    main()

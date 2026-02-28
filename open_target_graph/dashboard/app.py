import streamlit as st
import polars as pl
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from stmol import showmol
import py3Dmol
import requests

st.set_page_config(layout="wide", page_title="OpenTargetGraph")

st.title("🧬 OpenTargetGraph: AI-Driven Target Discovery")
st.markdown("""
This dashboard visualizes **Kinase targets** and their structural similarity using **ESM-2 Embeddings**. 
Instead of traditional sequence alignment, we use a **Protein Language Model** to capture deep semantic relationships between proteins.

#### 🤖 Model: ESM-2
**ESM-2** is a transformer-based model trained on millions of protein sequences. It converts a protein sequence into an embedding that encodes structural and functional properties.
""")

# --- Load Data ---
@st.cache_data
def load_data():
    # Load metadata and embeddings
    df_meta = pl.read_parquet("data/kinases.parquet")
    df_emb = pl.read_parquet("data/embeddings.parquet")
    
    # Join them on ID
    df = df_meta.join(df_emb, on="uniprot_id")
    return df

try:
    df = load_data()
    st.success(f"Loaded {len(df)} targets with embeddings.")
except Exception as e:
    st.error("Data not found! Did you run the Dagster pipeline? (Check 'data/' folder)")
    st.stop()

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Select a Target")
    selected_id = st.selectbox(
        "Choose a protein:", 
        df["uniprot_id"].to_list(),
        format_func=lambda x: f"{x} - {df.filter(pl.col('uniprot_id') == x)['protein_name'][0]}"
    )
    
    # Get details
    target_row = df.filter(pl.col("uniprot_id") == selected_id)
    seq = target_row["sequence"][0]
    
    st.text_area("Sequence", seq, height=100)
    
with col2:
    st.subheader("3D Structure Preview")
    # We use AlphaFoldDB to get the structure (predicted) for this ID
    # Note: ESMFold is another option, but AlphaFoldDB is easiest for a demo
    # Use the API to find the correct URL (handles versioning automatically)
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{selected_id}"
    api_response = requests.get(api_url)

    xyzview = py3Dmol.view(width=400, height=300)
    if api_response.ok and len(api_response.json()) > 0:
        pdb_url = api_response.json()[0]["pdbUrl"]
        pdb_response = requests.get(pdb_url)
        xyzview.addModel(pdb_response.text, "pdb")
        xyzview.setStyle({'cartoon':{'color':'spectrum'}})
        xyzview.zoomTo()
    else:
        st.warning(f"Structure not found for {selected_id}")

    showmol(xyzview, height=300, width=400)

st.divider()
st.subheader("2. Embedding Space (t-SNE)")
st.markdown("""
**What is this plot?**
We use **t-SNE** (t-Distributed Stochastic Neighbor Embedding) to project the 320-dimensional ESM-2 vectors down to 2D.
* **Points**: Each dot is a Kinase protein.
* **Proximity**: Points closer together are "semantically" similar in the eyes of the AI model.
""")

# Run t-SNE (Dimensionality Reduction)
# Note: In a real app, may be faster to pre-compute this and store it.
if st.button("Generate Plot"):
    with st.spinner("Projecting 320-dim vectors to 2D..."):
        # Extract embeddings matrix
        matrix = np.array(df["embedding"].to_list())
        
        # Reduce to 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
        projections = tsne.fit_transform(matrix)
        
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
            color="length", # Color by protein length as a proxy for complexity
            title="Protein Similarity Map (ESM-2 Latent Space)"
        )
        
        # Highlight selected point
        selected_point = plot_df.filter(pl.col("uniprot_id") == selected_id)
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
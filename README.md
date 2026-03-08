# OpenTargetGraph: AI-Driven Drug Discovery Platform

[![Dagster](https://img.shields.io/badge/Orchestration-Dagster-green)](https://dagster.io/)
[![Polars](https://img.shields.io/badge/Data-Polars-blue)](https://pola.rs/)
[![PyTorch](https://img.shields.io/badge/ML-PyTorch%20%2F%20ESM--2-red)](https://pytorch.org/)
[![Kubernetes](https://img.shields.io/badge/Infra-Kubernetes-blueviolet)](https://kubernetes.io/)

**OpenTargetGraph** is a cloud-native, end-to-end bioinformatics platform designed to identify and visualize potential drug targets using state-of-the-art Protein Language Models (PLMs). 

It demonstrates a modern **TechBio stack**, combining robust data engineering (Polars/Parquet), scalable orchestration (Dagster), and AI-driven structural biology (ESM-2 Embeddings) to bridge the gap between raw genomic data and actionable therapeutic insights.

## 🚀 High-Level Overview

This platform answers the question: *Which drug targets are structurally similar to known kinase inhibitors, based on deep learning embeddings rather than just sequence alignment?*
- Investigated targets: Known kinase inhibitors from UniProt.
- Investigated drugs: Bioactive molecules from ChEMBL.

1.  **Data Ingestion**: Automates the retrieval of high-value drug targets (e.g., Kinases) from **UniProt** and bioactive small molecules from **ChEMBL**.
2.  **AI Analysis**: Generates high-dimensional vector embeddings for protein sequences using Meta AI's **ESM-2 (Evolutionary Scale Modeling)** transformer.
3.  **Knowledge Graph**: Links targets to drugs in a relational schema, enabling complex queries about bioactivity and mechanism of action.
4.  **Visualization**: A **Streamlit** dashboard that offers:
    *   3D Protein Structure rendering (via Py3Dmol).
    *   An "Embedding Space" t-SNE projection to find novel clusters of similar targets.
    *   **Autonomous Research Assistant**: Deep-dive literature analysis via PubMed and LLM-driven research reports.
    *   **Semantic search for drug candidates** based on protein similarity.

📦 Project Structure
--------------------

```
├── open_target_graph/
│   ├── assets/             # Dagster Software-Defined Assets
│   │   ├── ingestion/      # ETL logic for UniProt/ChEMBL
│   │   └── modeling/       # PyTorch inference logic
│   └── dashboard/          # Streamlit frontend application
├── infra/                  # Pulumi IaC definitions
├── data/                   # Local storage for Parquet files (gitignored)
├── docker-compose.yml      # Docker Compose file for local development
├── Dockerfile.dagster      # Dockerfile for Dagster
├── Dockerfile.streamlit    # Dockerfile for Streamlit
└── pyproject.toml          # Python package and dependency management
```

## 🏗️ Architecture

The system follows a microservice-inspired architecture, orchestrated by Dagster and deployed on Kubernetes.

```mermaid
graph TD
    subgraph "Data Layer (Dagster + Polars)"
        A[UniProt API] -->|Fetch| B(Raw Parquet)
        C[ChEMBL API] -->|Fetch| B
        B -->|Clean & Join| D(Silver Tables)
    end

    subgraph "ML Layer (PyTorch)"
        D -->|Sequence| E[ESM-2 Transformer]
        E -->|Inference| F(Vector Embeddings)
    end

    subgraph "Storage & Serving"
        D --> G[(PostgreSQL)]
        F --> G
        G -.->|pgvector| H[Streamlit App]
        H --> I[PubMed API]
        H --> J[Gemini AI]
    end
```

# Developer documentation

### (Optional) Hugging Face Authentication 

The modeling pipeline downloads the `facebook/esm2...` model from the Hugging Face Hub. To avoid rate limits and enable faster downloads, you should use an access token.

1.  Create a free account on HuggingFace.co.
2.  Go to your **Access Tokens** and create a new token with `read` permissions.
3.  Create a `.env` file in the root of the project.
4.  Add your token to the `.env` file. Dagster will automatically load this for you.
    ```
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```
5.  Ensure `.env` is added to your `.gitignore` file to avoid committing secrets.

### (Optional) Gemini API Key

The dashboard uses the Gemini API for the research assistant. To use this feature, you need a Gemini API key.

1.  Create a free account on [Google AI Studio](https://aistudio.google.com/).
2.  Go to **Get API Key** and create a new API key.
3.  Add your API key to the `.env` file.
    ```
    GEMINI_API_KEY=your_gemini_api_key
    ```
4.  Ensure `.env` is added to your `.gitignore` file to avoid committing secrets.

## 🛠️ Local Setup (Current Status)

<details>

<summary>Manual Setup</summary>


### Prerequisites

*   Python 3.9+
*   [uv](https://github.com/astral-sh/uv): A fast Python package installer and resolver, used for environment management.

### 1. Installation

Clone the repository and create a virtual environment using `uv`.

```bash
uv venv
uv pip install -e ".[dev]"
```

### 2. Run the Data Pipeline (Dagster)

The project uses Dagster to orchestrate data fetching and ML model inference. Run the following command to launch the Dagster UI:

```bash
uv run dagster dev
```

This will start the Dagster UI, typically at http://localhost:3000

Navigate to the Dagster UI in your browser. Materialize the assets. This will execute the pipeline, download the data from UniProt and ChEMBL, generate embeddings, and save the results into the `data/` directory.

### 3. Run the Dashboard (Streamlit)

Once the data assets from the pipeline exist in the `data/` folder, you can launch the interactive Streamlit dashboard.

```bash
uv run streamlit run open_target_graph/dashboard/app.py
```

The application will now be running and accessible  at http://localhost:8501.

</details>

## 🐳 Docker Setup

To run the entire application stack including Dagster, PostgreSQL (with `pgvector`), and the Streamlit dashboard all at once:

1. Ensure Docker is installed and running.
2. Run the following command from the project root:
   ```bash
   docker compose up --build -d
   ```
3. Open the Dagster UI at http://localhost:3000 and materialize the assets. This will execute the pipeline, download the data from UniProt and ChEMBL, generate embeddings, and save the results into the `data/` directory.
4. Wait for the data ingestion to finish, then open Streamlit at http://localhost:8501.

To stop the application, run the following command:

```bash
docker compose down
```

## Testing

See manual setup above.

```bash
uv run pytest
```
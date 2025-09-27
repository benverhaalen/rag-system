# RAG Starter Skeleton

This repository is a blank canvas for building a Retrieval-Augmented Generation project. Only the minimal structure remains so you can decide how to implement each step.

## Structure
- `src/config.py` — Placeholder for environment and configuration handling.
- `src/ingestion.py` — Add document loading, chunking, and embedding logic.
- `src/vector_store.py` — Integrate ChromaDB collections and persistence.
- `src/retrieval.py` — Implement similarity search against stored embeddings.
- `src/generation.py` — Orchestrate prompt construction and model responses.
- `src/ui_streamlit.py` — Build the Streamlit interface when ready.
- `data/docs/` — Drop reference documents here before ingestion.
- `data/index/` — Persist your ChromaDB storage (kept empty for now).

## Getting Started
1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and populate secrets and paths.
4. Implement each module in `src/` as you flesh out the pipeline.

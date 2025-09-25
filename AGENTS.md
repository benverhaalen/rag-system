# Repository Guidelines

This repository hosts a minimal Streamlit-first RAG chatbot in Python. The FAISS MVP is meant to ship fast while exposing clear extension seams.

## Architecture & Workflow
Ingestion chunks Markdown and text files to ~800 tokens with overlap, embeds them through OpenAI, and stores vectors in FAISS. Queries retrieve the top passages, build a grounded prompt with inline citation markers, and call an OpenAI chat model for concise answers plus sources. Streamlit launches ingestion on first run, orchestrates retrieval, and presents expandable snippets for verification.

## Project Structure
- `src/` — production code by role (`ingest`, `index`, `retrieval`, `generation`, `ui`); central settings live in `src/config.py`.
- `tests/` — natural-language expectation files that describe behaviours and edge cases (no executable code yet).
- `data/docs/` — tiny Markdown samples checked into git.
- `data/index/` — FAISS index output; kept empty via `.gitkeep` and `.gitignore` rules.
- `scripts/` — automation hooks (e.g., `run_ingest.py`, dev helpers).

## Build & Run Commands
- `python -m venv venv` && `source venv/bin/activate` — prepare the environment.
- `pip install -r requirements.txt` — pull Streamlit, OpenAI, LangChain, FAISS, and tiktoken.
- `streamlit run src/ui/app.py` — launch the end-to-end experience.
- `python scripts/run_ingest.py` — placeholder CLI path for re-indexing outside the UI.

## Coding Style & Naming
Adopt PEP 8 with 4-space indentation, explicit type hints, lowercase_with_underscores for modules/functions, CapWords for classes, and UPPER_SNAKE_CASE for constants. Environment variables mirror `Settings` aliases (`OPENAI_API_KEY`, `MODEL_EMBED`, `MODEL_CHAT`, etc.). Use dependency injection to keep stages testable and comment only when behaviour is non-obvious.

## Testing Guidelines
Capture planned coverage in `tests/*.md` using plain-English assertions ("Assert that…"). Document expectations for chunk boundaries, FAISS persistence, retrieval ranking, prompt assembly, and UI rendering; these notes seed future automated tests.

## Commit & PR Guidelines
Write short imperative commit subjects, adding bodies only when nuance matters. Pull requests should summarize intent, link to issues or milestones, include Streamlit screenshots for UI changes, and confirm that documented expectations still hold. Never commit `.env` secrets; request review when retrieval logic or defaults shift.

## Environment & Secrets
`.env` powers typed `Settings`. Supply `OPENAI_API_KEY`, choose models, tune `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `TOP_K`, and `MAX_SNIPPETS_IN_PROMPT`, and set `INDEX_PATH` plus `DOCS_PATH`. Surface missing keys in the UI, but share placeholders securely instead of committing them.

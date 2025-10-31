# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system designed to process YouTube video transcripts, chunk them, store embeddings in ChromaDB, and enable semantic search for question answering.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY
```

## Architecture

### Core Pipeline

The RAG system follows this data flow:

1. **Ingestion** (`src/ingestion.py`): Fetches YouTube transcripts and chunks them
   - `get_id(url)`: Extracts YouTube video ID from URL using regex
   - `get_transcript(url)`: Retrieves raw transcript data via YouTube Transcript API
   - `chunk_transcript(transcript)`: Splits transcript into 800-character chunks with 200-character overlap using LangChain's RecursiveCharacterTextSplitter
   - Maintains timestamp mapping so each chunk knows its position in the video

2. **Embeddings** (`src/embeddings.py`): Stores chunked text with embeddings in ChromaDB
   - `init_chroma_client()`: Creates persistent ChromaDB client
   - `get_embedding_function()`: Configures OpenAI embedding function with API key
   - `get_collection(client, video_id)`: Creates/retrieves collection per video with OpenAI embeddings
   - `embed_and_store(chunks, video_id)`: Stores chunks with metadata (timestamp, video_id)
   - `get_collection_info(video_id)`: Retrieves collection metadata and statistics

3. **Retrieval** (`src/retrieval.py`): Semantic search against stored embeddings
   - `search_video(video_id, query, n_results)`: Performs similarity search and returns ranked chunks
   - `get_context_window(video_id, timestamp, window_seconds)`: Gets surrounding chunks for context
   - `list_available_videos()`: Lists all ingested video IDs

4. **Generation** (`src/generation.py`): RAG-based answer generation using GPT-4o-mini
   - `generate_answer(video_id, query, n_results)`: Full RAG pipeline with source citations
   - `generate_summary(video_id)`: Creates video summary from sampled chunks
   - `build_rag_prompt(query, context_chunks)`: Constructs prompts with retrieved context
   - `format_timestamp(seconds)`: Formats timestamps for display
   - `generate_youtube_link(video_id, timestamp)`: Creates timestamped YouTube URLs

### Key Design Patterns

**Timestamp Preservation**: The chunking process maintains a character-position-to-timestamp mapping, allowing each chunk to be traced back to its exact moment in the video. This is critical for citation and playback features.

**Collection Per Video**: Each YouTube video gets its own ChromaDB collection (named `video_{video_id}`), enabling isolation and easier management.

## Dependencies

- `streamlit`: UI framework (not yet implemented)
- `openai`: For embeddings (text-embedding-3-small) and LLM calls (GPT-4o-mini)
- `chromadb`: Vector database for embedding storage with persistent storage
- `langchain`: Text splitting utilities (RecursiveCharacterTextSplitter)
- `tiktoken`: Token counting for OpenAI models
- `youtube-transcript-api`: Transcript retrieval from YouTube
- `python-dotenv`: Environment variable loading from .env files

## Current State

**Fully Implemented**:
- YouTube transcript fetching with URL validation and error handling
- Intelligent text chunking with timestamp preservation (800 chars, 200 overlap)
- ChromaDB storage with persistent storage and OpenAI embeddings
- Semantic search with similarity scoring
- RAG-based question answering with source citations
- Video summarization
- Command-line interface for all operations
- Environment variable loading with python-dotenv
- Comprehensive error handling throughout

**Not Yet Implemented**:
- UI layer (Streamlit interface mentioned in README)
- Tests

## Usage

The system is operated via the command-line interface in `main.py`:

```bash
# Ingest a video
python main.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"

# Ask a question about a video
python main.py ask VIDEO_ID "What is the main topic?"

# Generate a summary
python main.py summarize VIDEO_ID

# List all ingested videos
python main.py list
```

## Implementation Details

### Retrieval (`src/retrieval.py`)
- `search_video()`: Semantic search using OpenAI embeddings, returns top N chunks with similarity scores
- `get_context_window()`: Retrieves chunks within a time window around a timestamp
- `list_available_videos()`: Lists all ingested video IDs

### Generation (`src/generation.py`)
- `generate_answer()`: Full RAG pipeline - retrieves context, builds prompt, calls GPT-4o-mini
- `generate_summary()`: Samples chunks across video and generates summary
- `format_timestamp()`: Converts seconds to MM:SS or HH:MM:SS format
- `generate_youtube_link()`: Creates YouTube URLs with timestamp parameters

### Main Interface (`main.py`)
- CLI with subcommands: ingest, ask, summarize, list
- Orchestrates the full pipeline from ingestion to answer generation
- Displays results with formatted timestamps and clickable YouTube links

## Key Features

- **Persistent Storage**: ChromaDB stores embeddings at `./data/index` (configurable via `CHROMA_DB_PATH`)
- **OpenAI Embeddings**: Uses `text-embedding-3-small` for semantic search
- **Source Citations**: Answers include source chunks with timestamps and YouTube links
- **Error Handling**: Comprehensive try-except blocks with descriptive error messages
- **Timestamp Links**: Generated answers include clickable YouTube links to exact moments in video

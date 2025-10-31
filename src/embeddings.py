import chromadb
from typing import Any, Dict, List
from chromadb.utils import embedding_functions
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# initialize chromadb client
def init_chroma_client():
    db_path = os.getenv('CHROMA_DB_PATH', './data/index')

    # ensure directory exists
    Path(db_path).mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=db_path)

def get_embedding_function() -> embedding_functions.OpenAIEmbeddingFunction:
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # return openai embedding function
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small"
    )

def get_collection(client, video_id: str):
    # get the embedding function for this collection
    embedding_fn = get_embedding_function()
    
    # create/retrieve collection with embeddings
    return client.get_or_create_collection(
        name=f"video_{video_id}",
        embedding_function=embedding_fn
    )

def embed_and_store(chunks: List[Dict[str, Any]], video_id: str):
    # initialize chromadb client
    client = init_chroma_client()
    collection = get_collection(client, video_id)

    # prepare data for chromadb
    ids = [f"{video_id}_chunk_{index}" for index in range(len(chunks))]
    documents = [str(chunk['text']) for chunk in chunks]
    metadatas: List[Dict[str, Any]] = [
        {
            "timestamp": float(chunk['timestamp']),
            "video_id": str(video_id)
        }
        for chunk in chunks
    ]

    # store embeddings in chromadb
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas  # type: ignore
    )

    return collection
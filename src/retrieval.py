from typing import Any, Dict, List
from dotenv import load_dotenv
from src.embeddings import init_chroma_client, get_embedding_function

load_dotenv()


def search_video(video_id: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    # get chromadb client and collection
    client = init_chroma_client()
    embedding_fn = get_embedding_function()
    collection = client.get_collection(
        name=f"video_{video_id}",
        embedding_function=embedding_fn  # type: ignore
    )

    # do semantic search
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # format results
    formatted_results: List[Dict[str, Any]] = []
    if results['documents']:
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': str(results['documents'][0][i]),
                'timestamp': float(results['metadatas'][0][i]['timestamp']),  # type: ignore
                'video_id': str(results['metadatas'][0][i]['video_id']),  # type: ignore
                'distance': float(results['distances'][0][i])  # type: ignore
            })

    return formatted_results


def get_context_window(video_id: str, timestamp: float, window_seconds: float = 30.0) -> List[Dict[str, Any]]:
    # get chromadb client and collection
    client = init_chroma_client()
    embedding_fn = get_embedding_function()
    collection = client.get_collection(
        name=f"video_{video_id}",
        embedding_function=embedding_fn  # type: ignore
    )

    # calculate time window
    start_time = max(0, timestamp - window_seconds)
    end_time = timestamp + window_seconds

    # get all chunks and filter by timestamp
    all_results = collection.get(include=["documents", "metadatas"])

    context_chunks: List[Dict[str, Any]] = []
    if all_results['documents']:
        for i in range(len(all_results['documents'])):
            chunk_timestamp = float(all_results['metadatas'][i]['timestamp'])  # type: ignore

            if start_time <= chunk_timestamp <= end_time:
                context_chunks.append({
                    'text': str(all_results['documents'][i]),
                    'timestamp': chunk_timestamp,
                    'video_id': str(all_results['metadatas'][i]['video_id'])  # type: ignore
                })

    # sort by timestamp
    context_chunks.sort(key=lambda x: float(x['timestamp']))

    return context_chunks


def list_available_videos() -> List[str]:
    # get all collections
    client = init_chroma_client()
    collections = client.list_collections()

    # extract video ids from collection names
    video_ids = []
    for collection in collections:
        if collection.name.startswith("video_"):
            video_id = collection.name.replace("video_", "")
            video_ids.append(video_id)

    return video_ids

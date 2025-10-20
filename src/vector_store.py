"""ChromaDB vector store for YouTube transcript chunks with timestamp metadata."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
import openai

# Handle both relative and absolute imports
try:
    from .ingestion import Chunk
except ImportError:
    from ingestion import Chunk


class YouTubeVectorStore:
    """Vector store for YouTube transcript chunks using ChromaDB."""
    
    def __init__(self, persist_directory: str = None, collection_name: str = "youtube_transcripts"):
        self.persist_directory = persist_directory or os.getenv("CHROMA_DB_PATH", "./data/index")
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._setup_client()
    
    def _setup_client(self):
        """Initialize ChromaDB client and collection."""
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "YouTube transcript chunks with timestamp metadata"}
        )
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add transcript chunks to the vector store with embeddings.
        
        Args:
            chunks: List of Chunk objects from the ingestion pipeline
        """
        if not chunks:
            return
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Extract text content for embedding
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings with OpenAI text-embedding-3-small...")
        embeddings = generate_embeddings(texts)
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents = []
        
        for i, chunk in enumerate(chunks):
            # Create unique ID combining video_id and segment span
            chunk_id = f"{chunk.video_id}_{chunk.segment_span[0]}_{chunk.segment_span[1]}"
            ids.append(chunk_id)
            
            # Create metadata matching resume specifications
            metadata = {
                "video_id": chunk.video_id,
                "start": chunk.start,
                "end": chunk.end,
                "url": chunk.jump_url,  # This includes the &t={seconds}s parameter
                "title": chunk.title,
                "display_range": chunk.display_range,  # e.g., "12:34–12:47"
                "segment_span_start": chunk.segment_span[0],
                "segment_span_end": chunk.segment_span[1],
                "char_len": chunk.char_len,
                "num_segments": chunk.num_segments
            }
            metadatas.append(metadata)
            documents.append(chunk.page_content)
        
        try:
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            print(f"Successfully added {len(chunks)} chunks to ChromaDB collection.")
            
        except Exception as e:
            print(f"Error adding chunks to collection: {e}")
            raise
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar transcript chunks based on query.
        
        Args:
            query: User's question/search query
            n_results: Number of similar chunks to return
            
        Returns:
            List of dicts with chunk content, metadata, and similarity scores
        """
        if not query.strip():
            return []
        
        print(f"Searching for chunks similar to: '{query}'")
        
        # Generate embedding for the query
        query_embedding = generate_embeddings([query])[0]
        
        try:
            # Search ChromaDB collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results for easy consumption
            formatted_results = []
            
            for i in range(len(results["documents"][0])):
                result = {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
                }
                formatted_results.append(result)
            
            print(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            print(f"Error searching collection: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored chunks."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using OpenAI text-embedding-3-small.
    
    Args:
        texts: List of text chunks to embed
        
    Returns:
        List of embedding vectors (lists of floats)
    """
    if not texts:
        return []
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        # Generate embeddings using text-embedding-3-small
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            encoding_format="float"
        )
        
        # Extract embedding vectors
        embeddings = [data.embedding for data in response.data]
        return embeddings
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    # Handle both relative and absolute imports
    try:
        from .ingestion import build_segments_from_csv, create_optimized_chunks
        from .config import config
    except ImportError:
        from ingestion import build_segments_from_csv, create_optimized_chunks
        from config import config
    
    # Validate configuration first
    if not config.validate():
        print("Please set up your .env file with OPENAI_API_KEY before testing.")
        exit(1)
    
    # Load sample data
    csv_path = Path("../data/docs/LPZh9BOjkQs_Large Language Models explained briefly.csv")
    if not csv_path.exists():
        # Try alternative path when running from src directory
        csv_path = Path("./data/docs/LPZh9BOjkQs_Large Language Models explained briefly.csv")
        
    if csv_path.exists():
        print("🔍 Loading and processing transcript data...")
        segments = build_segments_from_csv(csv_path)
        chunks = create_optimized_chunks(segments)
        print(f"Created {len(chunks)} chunks from {len(segments)} segments")
        
        # Initialize vector store
        print("\n📚 Initializing vector store...")
        store = YouTubeVectorStore()
        initial_stats = store.get_collection_stats()
        print(f"Vector store stats: {initial_stats}")
        
        # Test adding chunks
        if initial_stats["total_chunks"] == 0:
            print("\n⬆️  Adding chunks to vector store...")
            store.add_chunks(chunks)
            updated_stats = store.get_collection_stats()
            print(f"Updated stats: {updated_stats}")
        else:
            print(f"\n📋 Vector store already contains {initial_stats['total_chunks']} chunks")
        
        # Test search functionality
        print("\n🔍 Testing search functionality...")
        test_queries = [
            "What are language models?",
            "How do neural networks work?",
            "What is machine learning training?"
        ]
        
        for query in test_queries:
            print(f"\n Query: '{query}'")
            results = store.search_similar(query, n_results=3)
            
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                print(f"   {i}. Similarity: {result['similarity']:.3f}")
                print(f"      Time: {metadata['display_range']} | Jump: {metadata['url']}")
                print(f"      Text: {result['content'][:100]}...")
        
        print("\n✅ Vector store testing completed!")
        
    else:
        print("❌ Sample CSV not found. Please check the data directory.")

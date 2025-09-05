from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Document:
    """Represents a document in the system, separates away from file format constraints
    """
    content: str          # The actual text
    metadata: Dict[str, Any]  # Title, source, page numbers, etc
    doc_id: str          # Unique identifier
    chunks: List['Chunk'] = None  # Will be populated by chunker
    
@dataclass  
class Chunk:
    """
    A semantic unit of text from a document.
    Gets embedded and searched against.
    """
    content: str         # The chunk text
    embedding: List[float] = None  # Vector representation
    metadata: Dict[str, Any] = None  # Parent doc, position, etc.
    chunk_id: str = None

@dataclass
class SearchResult:
    """
    What is returned from a search query.
    Separates search logic from presentation logic.
    """
    chunk: Chunk
    score: float         # Similarity score
    context: str = None  # Surrounding text for better answers


# Abstract base classes define what each component MUST implement
# This is your contract - any implementation must fulfill this

class DocumentProcessor(ABC):
    """
    For anything that processes documents.
    """
    
    @abstractmethod
    def load_document(self, file_path: Path) -> Document:
        """Load a document from disk into the system"""
        pass
    
    @abstractmethod
    def chunk_document(self, document: Document, chunk_size: int = 1000) -> List[Chunk]:
        """Split document into semantic chunks"""
        pass


class EmbeddingService(ABC):
    """
    For embedding generation.
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch processing for efficiency"""
        pass


class VectorStore(ABC):
    """
    For vector storage.
    """
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Store chunks with their embeddings"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        """Find k most similar chunks"""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks from a document"""
        pass
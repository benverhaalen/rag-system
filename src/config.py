"""Configuration management for environment variables and application settings."""

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: str = ".env") -> None:
    """Load environment variables from .env file if it exists."""
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value


class Config:
    """Configuration class for RAG system settings."""
    
    def __init__(self):
        # Load .env file if it exists
        load_env_file()
        
        # OpenAI configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = "text-embedding-3-small"
        self.generation_model = "gpt-5-mini"
        
        # ChromaDB configuration
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./data/index")
        self.collection_name = "youtube_transcripts"
        
        # Application settings
        self.chunk_size = 750  # tokens
        self.chunk_overlap = 150  # tokens
        self.max_retrieval_results = 5
    
    def validate(self) -> bool:
        """Validate that required configuration is present."""
        if not self.openai_api_key:
            print("Error: OPENAI_API_KEY not found in environment variables.")
            print("   Please add your OpenAI API key to the .env file.")
            return False
        
        # Ensure chroma directory exists
        Path(self.chroma_db_path).mkdir(parents=True, exist_ok=True)
        
        print("Configuration validated successfully.")
        return True
    
    def display_settings(self) -> None:
        """Display current configuration settings."""
        print("RAG System Configuration:")
        print(f"   OpenAI API Key: {'Set' if self.openai_api_key else 'Missing'}")
        print(f"   Embedding Model: {self.embedding_model}")
        print(f"   Generation Model: {self.generation_model}")
        print(f"   ChromaDB Path: {self.chroma_db_path}")
        print(f"   Collection Name: {self.collection_name}")
        print(f"   Chunk Size: {self.chunk_size} tokens")
        print(f"   Chunk Overlap: {self.chunk_overlap} tokens")


# Global configuration instance
config = Config()


# Example usage and testing
if __name__ == "__main__":
    config.display_settings()
    
    if config.validate():
        print("\nAll configuration checks passed!")
    else:
        print("\nConfiguration validation failed. Please check your .env file.")

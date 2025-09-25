# src/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    embed_model: str = Field("text-embedding-3-small", alias="EMBED_MODEL")
    chat_model: str = Field("gpt-5-mini", alias="CHAT_MODEL")
    chunk_tokens: int = Field(800, alias="CHUNK_TOKENS")
    chunk_overlap: int = Field(120, alias="CHUNK_OVERLAP")

    # tells pydantic settings to read from .env automatically
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

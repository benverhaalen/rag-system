"""Configuration loader for environment-backed settings."""

from typing import Dict, Any


ENV_KEYS = (
    "OPENAI_API_KEY",
    "MODEL_EMBED",
    "MODEL_CHAT",
    "CHUNK_SIZE_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "TOP_K",
    "MAX_SNIPPETS_IN_PROMPT",
    "INDEX_PATH",
    "DOCS_PATH",
)


def get_settings(env: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Return a settings dictionary populated from the provided env mapping."""
    raise NotImplementedError("Settings retrieval is not yet implemented.")

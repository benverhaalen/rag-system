"""Filesystem loader for Markdown and text input documents."""

from pathlib import Path
from typing import List, Dict


SUPPORTED_EXTENSIONS = (".md", ".txt")


def load_documents(folder_path: Path) -> List[Dict[str, str]]:
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"load_documents: folder does not exist: {folder_path!r}.")
    
    candidates: List[Path] = []
    for fp in SUPPORTED_EXTENSIONS:
        candidates.extend(folder.rglob(fp))
        candidates.extend(folder.rglob(fp.upper()))
    

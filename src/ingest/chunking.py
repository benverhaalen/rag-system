"""Chunking utilities for token-bound document slicing."""

from typing import List, Tuple, Dict


def make_chunks(doc: Dict[str,str], chunk_size_tokens: int, chunk_overlap_tokens: int) -> List[Dict[str, object]]:
    
    # checking parameters
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be a positive integer.")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens cannot be negative.")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens"
         "(otherwise the window cannot advance).")
    
    doc_id: str = doc["doc_id"]
    title: str = doc["title"]
    source_path: str = doc["source_path"]
    raw_text: str = doc["raw_text"]

    # split into rough tokens
    words: List[str] = raw_text.split()
    
    if len(words) == 0:
        return []
    
    # need each chunk to know the exact position so the ui can later highlight the exact source
    word_starts: List[int] = []
    pos: int = 0
    for w in words:
        # find next occurence of word
        found_at = raw_text.find(w, pos)
        if found_at == -1:
            # if not found try from 0 start
            found_at = raw_text.find(w)
            if found_at == -1:
                # if still not found, approximate by using current pointer
                found_at = pos
        word_starts.append(found_at)
        # move pointer to the end of this word so the next search continues after it
        pos = found_at + len(w)
        
    # slide a fixed window over word indexes
    window: int = chunk_size_tokens
    step: int = chunk_size_tokens - chunk_overlap_tokens

    chunks: List[Dict[str, object]] = []
    # used to format a suffix
    chunk_counter: int = 0
    
    def _seq(n: int) -> str:
        return f"{n:04d}"
    
    # walk the list
    i: int = 0
    while i < len(words):
        # find the end index
        j: int = i + window
        if j > len(words):
            j = len(words)

        # extract the words for this chunk and join them back into text with spaces
        chunk_words: List[str] = words[i:j]
        chunk_text: str = " ".join(chunk_words)

        # estimate tokens
        token_estimate: int = len(chunk_words)

        # compute character span in original text:
        # start_char is the starting char offset of the first word in the chunk
        # end_char is the end offset of the last word in the chunk
        if len(chunk_words) > 0:
            start_char: int = word_starts[i]
            last_word_index: int = j - 1
            last_word_start: int = word_starts[last_word_index]
            last_word: str = words[last_word_index]
            end_char: int = last_word_start + len(last_word)
        else:
            # fall back for an empty slice
            start_char, end_char = 0, 0

        # build the chunk dict exactly as the rest of the app expects
        chunk_counter += 1
        chunk_id = f"{doc_id}:{_seq(chunk_counter)}"
        chunk: Dict[str, object] = {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "char_span": (start_char, end_char),
            "token_estimate": token_estimate,
            "metadata": {
                "doc_id": doc_id,
                "title": title,
                "source_path": source_path,
            },
        }
        chunks.append(chunk)

        # advance step
        i += step

        # break to avoid small end chunk
        if j == len(words):
            break

    # make sure at least one chunk
    if not chunks:
        # as a fallback emit one chunk with the entire text
        chunk_counter = 1
        chunk = {
            "chunk_id": f"{doc_id}:{_seq(chunk_counter)}",
            "text": raw_text,
            "char_span": (0, len(raw_text)),
            "token_estimate": len(words),
            "metadata": {
                "doc_id": doc_id,
                "title": title,
                "source_path": source_path,
            },
        }
        chunks.append(chunk)

    return chunks
from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float
    display_ts: str
    video_id: str
    title: str
    url: str
    source_path: Path
    
@dataclass(frozen=True)
class Chunk:
    page_content: str
    start: float
    end: float
    video_id: str
    title: str
    url: str
    source_path: Path
    segment_span: Tuple[int, int]
    char_len: int
    num_segments: int


def filename_to_metadata(csv_path: Path) -> Dict[str, object]:
    
    # path obj from filepath
    path = Path(csv_path).resolve()
    
    # filename
    filename = path.name
    
    # filename no ext
    stem = path.stem
    
    if "_" not in stem:
        raise ValueError(f"Filename must contain an underscore separating ID from TITLE")
    
    video_id, title = stem.split("_", 1)
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    return {
        "video_id": video_id,
        "title": title,
        "url": url,
        "path": path,
    }
    
def load_csv_rows(csv_path: Path) -> List[Tuple[str, float, float, str]]:
    path = Path(csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    
    rows: List[Tuple[str, float, float, str]] = []
    
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start = 1):
            # strip trailing \n and whitespace
            line = raw.rstrip("\n").strip()
            
            #skip blank lines
            if not line:
                continue
            
            parts = line.split(",", 3)
            
            if len(parts) < 4:
                raise ValueError(
                    f"Line {line_num}: expected 4 fields "
                    f"(display_ts,start,duration,text...), got {len(parts)}. Line: {raw!r}"
                )
            
            # get non content fields
            display_ts_raw = parts[0].strip()
            start_raw = parts[1].strip()
            duration_raw = parts[2].strip()
            
            # get text field
            text_raw = parts[3].strip()
            
            # raise error if not floats
            try:
                start = float(start_raw)
            except ValueError as e:
                raise ValueError(
                    f"Line {line_num}: 'start' must be a float (seconds). Got: {start_raw!r}"
                ) from e

            try:
                duration = float(duration_raw)
            except ValueError as e:
                raise ValueError(
                    f"Line {line_num}: 'duration' must be a float (seconds). Got: {duration_raw!r}"
                ) from e
                
            # error if duration 0 or less
            if duration <= 0:
                raise ValueError(
                    f"Line {line_num}: 'duration' should be > 0. Got: {duration}"
                )
            
            rows.append((display_ts_raw, start, duration, text_raw))
            
    return rows
    
def build_segments_from_csv(csv_path: Path) -> List[Segment]:

    # get id, title, url, path
    meta = filename_to_metadata(csv_path)
    
    video_id: str = meta["video_id"]
    title: str = meta["title"]
    url: str = meta["url"]
    source_path: Path = meta["path"]
    
    # load csv rows as tuples
    rows = load_csv_rows(source_path)
    
    segments: List[Segment] = []
    
    # iterate in order
    for idx, (display_ts, start, duration, text) in enumerate(rows):
        end = start + duration
        
        normalized_text = text.strip()
        
        seg = Segment(
            text=normalized_text,
            start=start,
            end=end,
            display_ts=display_ts,
            video_id=video_id,
            title=title,
            url=url,
            source_path=source_path,
        )
        segments.append(seg)
    
    return segments

# quick manual check
if __name__ == "__main__":
    CSV_EXAMPLE_PATH = Path("/Users/benverhaalen/rag-system/data/docs/LPZh9BOjkQs_Large Language Models explained briefly.csv")

    # extract filename metadata
    meta = filename_to_metadata(CSV_EXAMPLE_PATH)
    print("Metadata:", meta)

    # load rows, build segments
    segments = build_segments_from_csv(CSV_EXAMPLE_PATH)
    print(f"Loaded {len(segments)} segments.")

    # a few samples
    for s in segments[:3]:
        print(
            f"[{s.display_ts}] {s.text[:80]!r} "
            f"(start={s.start:.2f}s, end={s.end:.2f}s) → jump: {s.url}&t={int(s.start)}s"
        )
        
    
def _join_with_separator(parts: List[str], sep: str) -> Tuple[str, int]:
    joined = sep.join(parts)
    return joined, len(joined)

def _compute_segment_lengths_for_overlap(
    segments: List[Segment],
    i: int,
    j: int,
    sep: str,
) -> List[int]:
    per_seg_lengths: List[int] = []
    for k in range(i, j + 1):
        text_len = len(segments[k].text)
        if k == i:
            # First segment contributes only its text length (no leading separator).
            per_seg_lengths.append(text_len)
        else:
            # Subsequent segments contribute separator + text.
            per_seg_lengths.append(len(sep) + text_len)
    return per_seg_lengths
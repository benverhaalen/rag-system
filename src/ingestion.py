from youtube_transcript_api import YouTubeTranscriptApi
from typing import Dict, List
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

urlTest = "https://www.youtube.com/watch?v=jyLXcy5SGd8"
youtube_id_re = re.compile(r"(?:(?<=v=)|(?<=be/)|(?<=embed/)|(?<=v/)|(?<=V/)|(?<=\?v=)|(?<=&v=))(?P<id>[\w-]{11})")

def get_id(url: str) -> str:
    m = youtube_id_re.search(url)
    
    if m is None:
        raise ValueError(f"Could not extract Youtube ID from URL")
    
    return m.group(0)

def get_transcript(url: str) -> List[Dict]:
    # extract video id and fetch transcript
    video_id = get_id(url)

    # fetch transcript - youtube api handles validation
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    return fetched_transcript.to_raw_data()
    
def chunk_transcript(transcript: List[Dict]) -> List[Dict]:
    # build full text and character-to-timestamp mapping
    full_text = ""
    position_timestamp_map = []

    for segment in transcript:
        start_pos = len(full_text)
        segment_text = segment['text']

        # add space between segments if needed
        if full_text and not full_text.endswith(' '):
            full_text += ' '
            start_pos += 1

        full_text += segment_text
        end_pos = len(full_text)

        position_timestamp_map.append({
            'char_start': start_pos,
            'char_end': end_pos,
            'timestamp': segment['start'],
            'text': segment_text
        })

    # split text into chunks using langchain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    # map each chunk back to its timestamp
    chunks_with_timestamps = []
    for chunk in chunks:
        chunk_start = full_text.find(chunk)
        timestamp = find_timestamp(chunk_start, position_timestamp_map)

        chunks_with_timestamps.append({
            'text': chunk,
            'timestamp': timestamp
        })

    return chunks_with_timestamps

def find_timestamp(char_pos: int, positions: List[Dict]) -> float:
    for segment in positions:
        if segment['char_start'] <= char_pos < segment['char_end']:
            return segment['timestamp']

    # return last known timestamp if not found
    return positions[-1]['timestamp'] if positions else 0.0

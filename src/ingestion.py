from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscript, FetchedTranscriptSnippet
from typing import Any, Dict, List, Optional
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

urlTest = "https://www.youtube.com/watch?v=jyLXcy5SGd8"
youtube_id_re = re.compile(r"(?:(?<=v=)|(?<=be/)|(?<=embed/)|(?<=v/)|(?<=V/)|(?<=\?v=)|(?<=&v=))(?P<id>[\w-]{11})")

def get_id(url: str) -> str | None:
    m = youtube_id_re.search(url)
    return m.group(0) if m else None

def get_transcript(url: str) -> List[Dict]:
    video_id = get_id(url)
    if not video_id:
        raise ValueError("Not valid Youtube link")
    
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
        
    return fetched_transcript.to_raw_data()
    
# chunk youtube transcript using langchain
def chunk_transcript(transcript: List) -> List:
    # creating a full text str of the transcript and a list to map the text to timestamps
    full_text = ""
    position_timestamp_map = []
    
    # creating the map
    for segment in transcript:
        start_pos = len(full_text)
        segment_text = segment['text']
        
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
        
    # splitting full text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)
        
    # creating list of the split chunks mapped to the timestamp they appear
    chunks_with_timestamps = []
    
    for chunk in chunks:
        chunk_start = full_text.find(chunk)
        
        if chunk_start == -1:
            print("Could not find chunk")
            
        timestamp = find_timestamp(chunk_start, position_timestamp_map)
        
        chunks_with_timestamps.append({
            'text': chunk,
            'timestamp': timestamp
        })
        
    return chunks_with_timestamps

# given a char position, output the original transcript segment it belongs to
def find_timestamp(char_pos: int, positions: list) -> float:
    for segment in positions:
        if segment['char_start'] <= char_pos < segment['char_end']:
            return segment['timestamp']
        
    # return last known timestamp if not found
    return positions[-1]['timestamp'] if positions else 0.0

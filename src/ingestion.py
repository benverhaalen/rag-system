# get link

# save file as a csv into data/docs
from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscript, FetchedTranscriptSnippet
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import re
import csv

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
    

ft = get_transcript(urlTest)
print(ft[:3])

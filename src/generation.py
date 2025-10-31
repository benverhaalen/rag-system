from openai import OpenAI
from typing import Dict, List
import os
from dotenv import load_dotenv
from src.retrieval import search_video

load_dotenv()


def init_openai_client() -> OpenAI:
    api_key = os.getenv('OPENAI_API_KEY')

    # create and return the openai client
    return OpenAI(api_key=api_key)


def build_rag_prompt(query: str, context_chunks: List[Dict]) -> str:
    # system instruction
    prompt = "[ROLE]: You are an expert assistant that answers questions based on the exact information found in YouTube video transcripts.\n\n"

    # add context section header
    prompt += "[CONTEXT]: Here are relevant excerpts from the video transcript:\n\n"

    # iterate through each context chunk, add chunk num and text content
    for i, chunk in enumerate(context_chunks, 1):
        prompt += f"[Excerpt {i}] (at {chunk['timestamp']:.1f} seconds):\n"
        prompt += f"{chunk['text']}\n\n"

    # add the user's question
    prompt += f"Based on the above excerpts, please answer this question:\n{query}\n\n"

    # add instructions for the response format
    prompt += "[OUTPUT]: Provide a clear answer <250 words based on the transcript excerpts."
    prompt += "If you reference specific information, mention which excerpt it came from. "
    prompt += "If the excerpts don't contain enough information to answer the question, say so."

    # return the complete prompt
    return prompt


def generate_answer(video_id: str, query: str, n_results: int = 5, model: str = "gpt-5-mini") -> Dict:
    # retrieve relevant chunks
    context_chunks = search_video(video_id=video_id, query=query, n_results=n_results)

    # if there are no chunks found
    if not context_chunks:
        return {
            'answer': "I couldn't find any relevant information in this video to answer your question.",
            'sources': [],
            'model': model,
            'video_id': video_id
        }

    # build prompt and call openai
    prompt = build_rag_prompt(query, context_chunks)
    client = init_openai_client()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    # prepare source citations
    sources = [
        {
            'text': chunk['text'],
            'timestamp': chunk['timestamp'],
            'video_id': chunk['video_id'],
            'similarity_score': 1 - chunk['distance']
        }
        for chunk in context_chunks
    ]

    return {
        'answer': answer,
        'sources': sources,
        'model': model,
        'video_id': video_id,
        'query': query
    }


def generate_summary(video_id: str, model: str = "gpt-5-mini") -> Dict:
    # retrieve diverse chunks using a broad query
    sample_chunks = search_video(
        video_id=video_id,
        query="main topics discussed in this video",
        n_results=10
    )

    # if no chunks found
    if not sample_chunks:
        return {
            'summary': "No content available to summarize for this video.",
            'model': model,
            'video_id': video_id
        }

    # build summarization prompt
    prompt = "[ROLE]: You are a helpful assistant that summarizes YouTube video transcripts.\n\n"
    prompt += "[CONTEXT]: Here are excerpts from throughout the video:\n\n"

    for i, chunk in enumerate(sample_chunks, 1):
        prompt += f"[{chunk['timestamp']:.1f}s]: {chunk['text']}\n\n"

    prompt += "Based on these excerpts, provide a concise 1 paragraph summary of the main topics and key points discussed in this video. Maximum 7 sentences. "

    # call openai to generate summary
    client = init_openai_client()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=800  # gpt-5-mini uses max_completion_tokens instead of max_tokens
    )

    summary = response.choices[0].message.content

    return {
        'summary': summary,
        'model': model,
        'video_id': video_id,
        'chunks_used': len(sample_chunks)
    }


def format_timestamp(seconds: float) -> str:
    # convert seconds to integer to avoid decimals
    seconds = int(seconds)

    # calculate hours
    hours = seconds // 3600

    # calculate remaining minutes after removing hours
    minutes = (seconds % 3600) // 60

    # calculate remaining seconds after removing hours and minutes
    secs = seconds % 60

    # format based on whether video is over an hour
    if hours > 0:
        # use HH:MM:SS
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        # use MM:SS
        return f"{minutes}:{secs:02d}"


def generate_youtube_link(video_id: str, timestamp: float) -> str:
    # convert timestamp to integer seconds
    timestamp_int = int(timestamp)

    # construct youtube url with timestamp
    return f"https://www.youtube.com/watch?v={video_id}&t={timestamp_int}s"

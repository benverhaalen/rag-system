#!/usr/bin/env python3
"""
Main entry point for the YouTube Transcript RAG System.
Provides a command-line interface to ingest videos and ask questions.
"""

# import argument parser for cli interface
import argparse
# import sys for system exit codes
import sys
# import the ingestion pipeline functions
from src.ingestion import get_transcript, chunk_transcript, get_id
# import the embedding storage function
from src.embeddings import embed_and_store
# import the retrieval functions
from src.retrieval import list_available_videos
# import the generation functions
from src.generation import generate_answer, generate_summary, format_timestamp, generate_youtube_link


def ingest_video(url: str) -> None:
    """
    Ingest a YouTube video: fetch transcript, chunk, and store embeddings.

    Args:
        url: YouTube video URL

    Raises:
        SystemExit: If ingestion fails
    """
    try:
        # print status message to user
        print(f"Ingesting video from URL: {url}")

        # fetch the transcript from youtube (validates url internally)
        print("Fetching transcript...")
        transcript = get_transcript(url)

        # extract video id for display and storage
        video_id = get_id(url)
        print(f"Video ID: {video_id}")

        # print number of segments retrieved
        print(f"Retrieved {len(transcript)} transcript segments")

        # step 3: chunk the transcript while preserving timestamps
        print("Chunking transcript...")
        # split transcript into semantic chunks
        chunks = chunk_transcript(transcript)

        # print number of chunks created
        print(f"Created {len(chunks)} chunks")

        # step 4: embed and store chunks in chromadb
        print("Generating embeddings and storing in ChromaDB...")
        # store chunks with openai embeddings
        collection = embed_and_store(chunks, video_id)

        # print success message
        print(f"Successfully ingested video {video_id}!")
        # print collection name for reference
        print(f"Collection: {collection.name}")
        # print final chunk count
        print(f"Total chunks stored: {len(chunks)}")

    # catch any errors during ingestion
    except Exception as e:
        # print error message
        print(f"Error during ingestion: {str(e)}")
        # exit with error code
        sys.exit(1)


def ask_question(video_id: str, question: str, n_results: int = 5) -> None:
    """
    Ask a question about a video and display the answer with sources.

    Args:
        video_id: YouTube video ID
        question: Question to ask about the video
        n_results: Number of context chunks to retrieve

    Raises:
        SystemExit: If question answering fails
    """
    try:
        # print status message
        print(f"Searching video {video_id} for: '{question}'")
        # add separator line
        print("-" * 80)

        # generate answer using rag pipeline
        result = generate_answer(
            video_id=video_id,  # which video to query
            query=question,  # user's question
            n_results=n_results  # how many context chunks to use
        )

        # print the generated answer
        print("\nAnswer:")
        print(result['answer'])

        # print sources section header
        print("\n" + "-" * 80)
        print("Sources:")

        # iterate through each source chunk
        for i, source in enumerate(result['sources'], 1):
            # print source number
            print(f"\n[Source {i}]")
            # format and print timestamp
            print(f"Time: {format_timestamp(source['timestamp'])}")
            # generate and print youtube link with timestamp
            print(f"Link: {generate_youtube_link(video_id, source['timestamp'])}")
            # print similarity score as percentage
            print(f"Relevance: {source['similarity_score']:.1%}")
            # print the source text
            print(f"Text: {source['text'][:200]}...")  # show first 200 chars

    # catch any errors during question answering
    except Exception as e:
        # print error message
        print(f"Error answering question: {str(e)}")
        # exit with error code
        sys.exit(1)


def summarize_video(video_id: str) -> None:
    """
    Generate and display a summary of a video.

    Args:
        video_id: YouTube video ID to summarize

    Raises:
        SystemExit: If summarization fails
    """
    try:
        # print status message
        print(f"Generating summary for video {video_id}...")
        # add separator line
        print("-" * 80)

        # generate summary using rag pipeline
        result = generate_summary(video_id=video_id)

        # print the summary
        print("\nSummary:")
        print(result['summary'])

        # print metadata
        print("\n" + "-" * 80)
        print(f"Model used: {result['model']}")
        print(f"Chunks analyzed: {result['chunks_used']}")

    # catch any errors during summarization
    except Exception as e:
        # print error message
        print(f"Error generating summary: {str(e)}")
        # exit with error code
        sys.exit(1)


def list_videos() -> None:
    """
    List all ingested videos.

    Raises:
        SystemExit: If listing fails
    """
    try:
        # get list of all available video ids
        video_ids = list_available_videos()

        # check if any videos have been ingested
        if not video_ids:
            # print message if no videos found
            print("No videos have been ingested yet.")
            # return early
            return

        # print header
        print(f"Found {len(video_ids)} ingested video(s):")
        # add separator
        print("-" * 80)

        # iterate through each video id
        for video_id in video_ids:
            # print video id and youtube url
            print(f"\nVideo ID: {video_id}")
            print(f"URL: https://www.youtube.com/watch?v={video_id}")

    # catch any errors during listing
    except Exception as e:
        # print error message
        print(f"Error listing videos: {str(e)}")
        # exit with error code
        sys.exit(1)


def main():
    """
    Main function that parses arguments and routes to appropriate command.
    """
    # create argument parser with description
    parser = argparse.ArgumentParser(
        description="YouTube Transcript RAG System - Ingest videos and ask questions"
    )

    # create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='command',  # store command name in this attribute
        help='Available commands',  # help text
        required=True  # command is required
    )

    # subcommand: ingest
    # used to ingest a new youtube video
    ingest_parser = subparsers.add_parser(
        'ingest',  # command name
        help='Ingest a YouTube video'  # help text
    )
    # add url argument for ingest command
    ingest_parser.add_argument(
        'url',  # argument name
        type=str,  # expect string type
        help='YouTube video URL'  # help text
    )

    # subcommand: ask
    # used to ask questions about ingested videos
    ask_parser = subparsers.add_parser(
        'ask',  # command name
        help='Ask a question about a video'  # help text
    )
    # add video_id argument
    ask_parser.add_argument(
        'video_id',  # argument name
        type=str,  # expect string type
        help='YouTube video ID'  # help text
    )
    # add question argument
    ask_parser.add_argument(
        'question',  # argument name
        type=str,  # expect string type
        help='Question to ask about the video'  # help text
    )
    # add optional n_results argument
    ask_parser.add_argument(
        '--n-results',  # argument flag
        type=int,  # expect integer type
        default=5,  # default value
        help='Number of context chunks to retrieve (default: 5)'  # help text
    )

    # subcommand: summarize
    # used to generate video summaries
    summary_parser = subparsers.add_parser(
        'summarize',  # command name
        help='Generate a summary of a video'  # help text
    )
    # add video_id argument
    summary_parser.add_argument(
        'video_id',  # argument name
        type=str,  # expect string type
        help='YouTube video ID'  # help text
    )

    # subcommand: list
    # used to list all ingested videos
    list_parser = subparsers.add_parser(
        'list',  # command name
        help='List all ingested videos'  # help text
    )

    # parse command line arguments
    args = parser.parse_args()

    # route to appropriate function based on command
    if args.command == 'ingest':
        # call ingest function with url
        ingest_video(args.url)

    elif args.command == 'ask':
        # call ask function with video_id, question, and n_results
        ask_question(args.video_id, args.question, args.n_results)

    elif args.command == 'summarize':
        # call summarize function with video_id
        summarize_video(args.video_id)

    elif args.command == 'list':
        # call list function
        list_videos()


# entry point when script is run directly
if __name__ == "__main__":
    # call main function
    main()

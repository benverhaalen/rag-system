"""Generation pipeline for YouTube transcript RAG system using GPT-5-mini."""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai

# Handle both relative and absolute imports
try:
    from .retrieval import YouTubeRetriever, RetrievalResult, create_prompt_template
    from .config import config
except ImportError:
    from retrieval import YouTubeRetriever, RetrievalResult, create_prompt_template
    from config import config


@dataclass
class GenerationResponse:
    """Container for generated response with metadata."""
    answer: str
    sources: List[RetrievalResult]
    query: str
    model_used: str
    tokens_used: Optional[int] = None


class YouTubeRAGGenerator:
    """Generation system using GPT-5-mini for YouTube transcript QA."""
    
    def __init__(self, retriever: Optional[YouTubeRetriever] = None):
        self.retriever = retriever or YouTubeRetriever()
        self.model = config.generation_model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_answer(self, query: str, n_retrieval_results: Optional[int] = None) -> GenerationResponse:
        """
        Generate an answer to a query using RAG pipeline.
        
        Args:
            query: User's question
            n_retrieval_results: Number of chunks to retrieve for context
            
        Returns:
            GenerationResponse with answer and source information
        """
        # Step 1: Retrieve relevant context
        retrieval_results = self.retriever.retrieve(query, n_retrieval_results)
        
        if not retrieval_results:
            return GenerationResponse(
                answer="I couldn't find any relevant information in the video transcripts to answer your question.",
                sources=[],
                query=query,
                model_used=self.model
            )
        
        # Step 2: Format context for LLM
        context = self.retriever.format_context_for_llm(retrieval_results)
        
        # Step 3: Create prompt template
        prompt = create_prompt_template(query, context)
        
        # Step 4: Generate response using GPT-5-mini
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            return GenerationResponse(
                answer=answer,
                sources=retrieval_results,
                query=query,
                model_used=self.model,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            return GenerationResponse(
                answer=f"Sorry, I encountered an error generating the response: {str(e)}",
                sources=retrieval_results,
                query=query,
                model_used=self.model
            )
    
    def format_response_for_ui(self, response: GenerationResponse) -> Dict[str, Any]:
        """
        Format generation response for Streamlit UI display.
        
        Args:
            response: GenerationResponse object
            
        Returns:
            Dictionary formatted for UI consumption
        """
        # Format sources with jump links
        sources_formatted = []
        for i, source in enumerate(response.sources, 1):
            source_dict = {
                "number": i,
                "title": source.title,
                "time_range": source.display_range,
                "content_preview": source.content[:150] + "..." if len(source.content) > 150 else source.content,
                "jump_url": source.jump_url,
                "similarity": source.similarity_score
            }
            sources_formatted.append(source_dict)
        
        return {
            "answer": response.answer,
            "query": response.query,
            "sources": sources_formatted,
            "model_info": {
                "model": response.model_used,
                "tokens_used": response.tokens_used
            },
            "source_count": len(response.sources)
        }


def create_rag_pipeline() -> YouTubeRAGGenerator:
    """Create a complete RAG pipeline instance."""
    return YouTubeRAGGenerator()


# Example usage and testing
if __name__ == "__main__":
    print("🤖 Testing YouTube RAG Generation System")
    print("=" * 50)
    
    # Validate configuration
    if not config.validate():
        print("❌ Please set up your .env file with OPENAI_API_KEY before testing.")
        exit(1)
    
    # Initialize RAG pipeline
    rag_generator = create_rag_pipeline()
    
    # Check if we have content in vector store
    retrieval_stats = rag_generator.retriever.get_retrieval_stats()
    print(f"Vector Store Stats: {retrieval_stats}")
    
    if retrieval_stats["total_chunks_available"] == 0:
        print("⚠️  No chunks in vector store. Run vector_store.py first to add content.")
    else:
        # Test generation with sample queries
        test_queries = [
            "What are language models and how do they work?",
            "How are neural networks trained?",
            "What makes large language models different?"
        ]
        
        for query in test_queries:
            print(f"\n❓ Question: '{query}'")
            print("-" * 50)
            
            # Generate answer
            response = rag_generator.generate_answer(query)
            
            print(f"🤖 Answer:\n{response.answer}\n")
            
            if response.sources:
                print(f"📚 Sources ({len(response.sources)} chunks):")
                for i, source in enumerate(response.sources, 1):
                    print(f"{i}. {source.display_range} | Similarity: {source.similarity_score:.3f}")
                    print(f"   Jump to: {source.jump_url}")
                    print(f"   Preview: {source.content[:100]}...\n")
            
            if response.tokens_used:
                print(f"📊 Tokens used: {response.tokens_used}")
            
            print("-" * 50)
    
    print("\n✅ RAG generation pipeline testing completed!")

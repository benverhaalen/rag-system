import hashlib
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from interfaces import Document, Chunk, DocumentProcessor

class PDFDocumentProcessor(DocumentProcessor):
    """_summary_
    Implementation for PDF processing
    Args:
        DocumentProcessor (_type_): _description_
    """
    
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    def load_document(self, file_path: Path) -> Document:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found.")
        
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Expected PDF file type")
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # extract text and include page nums
                text_content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text_content += f"\n[Page {page_num + 1}]\n{page_text}"
                
                # generate doc id by hashing content into hexadecimal
                doc_id = hashlib.md5(text_content.encode()).hexdigest()
                
                # get metadata
                metadata = {
                    'source': str(file_path),
                    'filename': file_path.name,
                    'page_count': len(pdf_reader.pages),
                    'doc_id': doc_id
                }
                
                return Document(
                    content=text_content,
                    metadata=metadata,
                    doc_id=doc_id
                )
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF {file_path}: {str(e)}")
        
    def chunk_document(self, document: Document, chunk_size: int = None) -> List[Chunk]:
        """
        Split document into overlapping chunks using sliding window
        """
        if chunk_size:
            self.chunk_size = chunk_size
        
        # initialize chunk output list and get content from document
        chunks = []
        text = document.content
        
        # simple chunking based on looking at end of sentences
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # makes sure the end is not larger than doc length
            end = min(start + self.chunk_size, len(text))
            
            # break at sentence boundary
            if end < len(text):
                # make sure the sentence end is near the end of chunk
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + self.chunk_size // 2:
                    end = last_period + 1
            
            # saves chunk and gets rid of extra formatting
            chunk_text = text[start:end].strip()
            
            # skip empty chunks
            if chunk_text:  
                # keep metadata for later
                chunk_metadata = {
                    'doc_id': document.doc_id,
                    'chunk_index': chunk_index,
                    'start_char': start,
                    'end_char': end,
                    'source': document.metadata['source']
                }
                
                # get id to track chunk
                chunk_id = f"{document.doc_id}_{chunk_index}"
                
                # add new chunk object to the list
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        chunk_id=chunk_id
                    )
                )
                chunk_index += 1
            
            # move forward with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        # update document with its chunks
        document.chunks = chunks
        return chunks
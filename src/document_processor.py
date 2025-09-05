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
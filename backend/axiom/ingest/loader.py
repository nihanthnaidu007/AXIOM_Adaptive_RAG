"""AXIOM Document Loader - PDF and Text Chunking (Fully Implemented)."""

import hashlib
from typing import List, Dict, Optional

import tiktoken
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

_tiktoken_enc = tiktoken.get_encoding("cl100k_base")


class DocumentChunker:
    """
    Document chunker using pdfplumber for PDFs and sliding window for text.
    Preserves paragraph and sentence boundaries.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,       # tokens (tiktoken cl100k_base)
        chunk_overlap: int = 64,      # overlap tokens between adjacent chunks
        min_chunk_size: int = 100     # discard chunks shorter than this (tokens)
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            min_chunk_size: Minimum chunk size to keep (tokens)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def load_pdf(self, file_path: str) -> List[Dict]:
        """
        Load PDF via pdfplumber.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of {page_num, text} dicts
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required for PDF loading. Install with: pip install pdfplumber")
        
        pages = []
        
        with pdfplumber.open(file_path) as pdf:
            # Detect repeated headers/footers
            first_lines = []
            last_lines = []
            
            for page in pdf.pages:
                text = page.extract_text() or ""
                lines = text.split('\n')
                if lines:
                    first_lines.append(lines[0] if lines else "")
                    last_lines.append(lines[-1] if lines else "")
            
            # Find repeated headers/footers (appear on >50% of pages)
            header_threshold = len(pdf.pages) * 0.5
            repeated_headers = set()
            repeated_footers = set()
            
            for line in set(first_lines):
                if first_lines.count(line) > header_threshold:
                    repeated_headers.add(line)
            
            for line in set(last_lines):
                if last_lines.count(line) > header_threshold:
                    repeated_footers.add(line)
            
            # Extract text from each page, removing headers/footers
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                lines = text.split('\n')
                
                # Remove repeated headers/footers
                if lines and lines[0] in repeated_headers:
                    lines = lines[1:]
                if lines and lines[-1] in repeated_footers:
                    lines = lines[:-1]
                
                cleaned_text = '\n'.join(lines).strip()
                if cleaned_text:
                    pages.append({
                        "page_num": page_num,
                        "text": cleaned_text
                    })
        
        return pages
    
    def load_text(self, text: str, source: str) -> List[Dict]:
        """
        Load raw text string directly.
        
        Args:
            text: Raw text content
            source: Source identifier
            
        Returns:
            List of {page_num, text} dicts (single page)
        """
        return [{"page_num": 1, "text": text.strip()}]
    
    def chunk(self, pages: List[Dict], source: str = "unknown") -> List[Dict]:
        """
        Sliding window chunking with overlap, respecting sentence boundaries.
        
        Args:
            pages: List of {page_num, text} dicts
            source: Source identifier for the document
            
        Returns:
            List of {chunk_id, source, content, chunk_index, token_count} dicts
        """
        # Combine all pages into one text
        full_text = "\n\n".join(page["text"] for page in pages)
        
        # Split into sentences
        sentences = self._split_into_sentences(full_text)
        
        chunks = []
        current_chunk: List[str] = []
        current_token_count = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = len(_tiktoken_enc.encode(sentence))
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_token_count + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_token_count = len(_tiktoken_enc.encode(chunk_text))
                
                if chunk_token_count >= self.min_chunk_size:
                    chunk_id = self._generate_chunk_id(source, chunk_index)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source": source,
                        "content": chunk_text,
                        "chunk_index": chunk_index,
                        "token_count": chunk_token_count
                    })
                    chunk_index += 1
                
                # Start new chunk with overlap: keep trailing sentences up to chunk_overlap tokens
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                for sent in reversed(current_chunk):
                    sent_tokens = len(_tiktoken_enc.encode(sent))
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break
                current_chunk = overlap_sentences
                current_token_count = overlap_tokens
            
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_token_count = len(_tiktoken_enc.encode(chunk_text))
            if chunk_token_count >= self.min_chunk_size:
                chunk_id = self._generate_chunk_id(source, chunk_index)
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": source,
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "token_count": chunk_token_count
                })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK sent_tokenize."""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_chunk_id(self, source: str, chunk_index: int) -> str:
        """Generate a unique chunk ID from source and index."""
        content = f"{source}:{chunk_index}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()
        return hash_digest[:12]

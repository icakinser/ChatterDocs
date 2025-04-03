import logging
from datetime import datetime
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    BSHTMLLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rake_nltk import Rake
from typing import List, Union
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def enrich_metadata(text: str, file_path: str, max_keywords: int = 5) -> dict:
    """Extract enhanced metadata from text and file.
    
    Args:
        text: Text content to analyze
        file_path: Path to source file
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        Dictionary containing enriched metadata including:
        - Keywords
        - Text statistics
        - File metadata
        - Content features
    """
    import re
    from datetime import datetime
    
    # Extract keywords
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()[:max_keywords]
    
    # Extract potential author/date from text
    author_match = re.search(r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
    date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(20\d{2})', text)
    
    # Get file info
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    file_type = {
        '.pdf': 'PDF',
        '.txt': 'Text',
        '.docx': 'Word',
        '.html': 'HTML',
        '.htm': 'HTML'
    }.get(file_ext, 'Unknown')
    
    # Content features
    sentences = re.split(r'[.!?]', text)
    avg_sentence_len = sum(len(s.split()) for s in sentences)/len(sentences) if sentences else 0
    
    return {
        'keywords': keywords,
        'length': len(text),
        'word_count': len(text.split()),
        'file_name': file_name,
        'file_type': file_type,
        'author': author_match.group(1) if author_match else None,
        'date': date_match.group() if date_match else None,
        'avg_sentence_length': round(avg_sentence_len, 1),
        'processed_at': datetime.now().isoformat()
    }

def preprocess_text(text: str) -> str:
    """Clean and normalize text before processing.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text with:
        - Extra whitespace removed
        - Multiple newlines collapsed
        - Common artifacts removed
    """
    import re
    
    # Remove non-breaking spaces and other special whitespace
    text = re.sub(r'[\xa0\u200b\u200c\u200d]', ' ', text)
    
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Remove common PDF/HTML artifacts
    text = re.sub(r'-\n', '', text)  # Handle hyphenated line breaks
    text = re.sub(r'\[.*?\]', '', text)  # Remove citations like [1]
    text = re.sub(r'\(.*?\)', '', text)  # Remove parentheticals
    
    return text.strip()

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
import traceback

def batch_process(file_paths: List[str], chunk_size: int = 512, chunk_overlap: int = 50, 
                 max_workers: int = 4) -> Dict[str, Any]:
    """Process multiple documents in parallel.
    
    Args:
        file_paths: List of document file paths to process
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary containing:
        - results: List of successfully processed documents
        - errors: List of failed documents with error info
        - stats: Processing statistics
    """
    results = []
    errors = []
    stats = {
        'total': len(file_paths),
        'processed': 0,
        'failed': 0,
        'chunks_generated': 0
    }
    
    def process_single(file_path):
        try:
            chunks = load_and_chunk(file_path, chunk_size, chunk_overlap)
            return {
                'file_path': file_path,
                'chunks': chunks,
                'success': True
            }
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            return {
                'file_path': file_path,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }
    
    logger.info(f"Starting batch processing of {len(file_paths)} documents with {max_workers} workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, fp): fp for fp in file_paths}
        
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                if result['success']:
                    results.extend(result['chunks'])
                    stats['chunks_generated'] += len(result['chunks'])
                    stats['processed'] += 1
                else:
                    errors.append({
                        'file_path': file_path,
                        'error': result['error']
                    })
                    stats['failed'] += 1
                
                logger.info(f"Progress: {stats['processed'] + stats['failed']}/{stats['total']} "
                          f"({(stats['processed'] + stats['failed'])/stats['total']:.0%})")
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                errors.append({
                    'file_path': file_path,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                stats['failed'] += 1
    
    logger.info(f"Batch processing complete. Success: {stats['processed']}, "
              f"Failed: {stats['failed']}, Total chunks: {stats['chunks_generated']}")
    return {
        'results': results,
        'errors': errors,
        'stats': stats
    }

def load_and_chunk(file_path: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[dict]:
    """Load and chunk documents from various file formats.
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of document chunks with metadata
    """
    logger.info(f"Starting document processing for: {file_path}")
    # Determine loader based on file extension
    ext = os.path.splitext(file_path)[1].lower()
    logger.debug(f"Detected file extension: {ext}")
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path)
    elif ext == '.docx':
        loader = Docx2txtLoader(file_path)
    elif ext in ('.html', '.htm'):
        try:
            loader = BSHTMLLoader(file_path)
        except:
            loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Load, preprocess and split documents
    logger.info("Loading document content")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} document sections")
    
    # Preprocess document content
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    logger.info(f"Chunking documents with size {chunk_size} and overlap {chunk_overlap}")
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from document")
    
    logger.info("Enriching chunks with metadata")
    enriched_chunks = []
    for i, chunk in enumerate(chunks, 1):
        try:
            metadata = chunk.metadata.copy()
            metadata.update(enrich_metadata(chunk.page_content, file_path))
            enriched_chunks.append({
                'text': chunk.page_content,
                'metadata': metadata
            })
            if i % 10 == 0 or i == len(chunks):
                logger.debug(f"Processed {i}/{len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {str(e)}")
            continue
    
    logger.info(f"Finished processing {file_path} - generated {len(enriched_chunks)} enriched chunks")
    return enriched_chunks

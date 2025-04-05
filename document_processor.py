import logging
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
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

def enrich_metadata(text: str, file_path: str, max_keywords: int = 5, cluster_id: int = None) -> dict:
    """Extract enhanced metadata from text and file.
    
    Args:
        text: Text content to analyze
        file_path: Path to source file
        max_keywords: Maximum number of keywords to extract
        cluster_id: Optional cluster assignment ID
        
    Returns:
        Dictionary containing enriched metadata including:
        - Keywords
        - Text statistics
        - File metadata
        - Content features
        - Cluster ID (if provided)
    """
    import re
    from datetime import datetime
    
    # Extract keywords with multiple fallback options
    keywords = []
    try:
        # Try RAKE first
        try:
            r = Rake()
            r.extract_keywords_from_text(text)
            keywords = r.get_ranked_phrases()[:max_keywords]
        except Exception as rake_error:
            logger.warning(f"RAKE keyword extraction failed: {str(rake_error)}")
            # Fallback 1: Try TF-IDF style extraction
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(stop_words='english', max_features=max_keywords)
                tfidf = vectorizer.fit_transform([text])
                keywords = vectorizer.get_feature_names_out().tolist()
            except Exception as tfidf_error:
                logger.warning(f"TF-IDF extraction failed: {str(tfidf_error)}")
                # Fallback 2: Simple word frequency approach
                try:
                    from nltk.tokenize import word_tokenize
                    from nltk.corpus import stopwords
                    from collections import Counter
                    words = [word.lower() for word in word_tokenize(text) 
                            if word.isalpha() and len(word) > 2 
                            and word.lower() not in stopwords.words('english')]
                    keywords = [w for w, _ in Counter(words).most_common(max_keywords)]
                except Exception as nltk_error:
                    logger.warning(f"Basic word extraction failed: {str(nltk_error)}")
                    # Final fallback: Simple split
                    keywords = list(set(text.lower().split()))[:max_keywords]
    except Exception as e:
        logger.error(f"All keyword extraction methods failed: {str(e)}")
        keywords = ["unknown"]  # Fallback minimal value
    
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
    
    try:
        metadata = {
            'keywords': keywords if keywords else [],  # Ensure empty list if no keywords
            'length': len(text),
            'word_count': len(text.split()),
            'file_name': file_name,
            'file_type': file_type,
            'author': author_match.group(1) if author_match else None,
            'date': date_match.group() if date_match else None,
            'avg_sentence_length': round(avg_sentence_len, 1) if sentences else 0,
            'processed_at': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating metadata: {str(e)}")
        # Fallback minimal metadata
        metadata = {
            'keywords': [],
            'length': len(text),
            'word_count': len(text.split()),
            'file_name': file_name,
            'file_type': file_type,
            'processed_at': datetime.now().isoformat()
        }
    if cluster_id is not None:
        metadata['cluster_id'] = cluster_id
    return metadata

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

def cluster_documents(
    chunks: List[dict],
    n_clusters: int = 5,
    algorithm: str = "kmeans",
    min_samples: int = 2,
    eps: float = 0.5
) -> Dict[str, Any]:
    """Cluster document chunks based on their embeddings.
    
    Args:
        chunks: List of document chunks with text and metadata
        n_clusters: Number of clusters (for KMeans)
        algorithm: "kmeans" or "dbscan"
        min_samples: Minimum samples per cluster (for DBSCAN)
        eps: Maximum distance between samples (for DBSCAN)
        
    Returns:
        Dictionary containing:
        - labels: Cluster assignments for each chunk
        - metrics: Clustering quality metrics
        - algorithm: Algorithm used
        - params: Parameters used
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    logger.info(f"Clustering {len(chunks)} documents using {algorithm}")
    
    # Generate embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            if 'text' in chunk and chunk['text']:
                texts.append(chunk['text'])
            elif 'page_content' in chunk and chunk['page_content']:  # Handle LangChain format
                texts.append(chunk['page_content'])
            else:
                logger.warning(f"Skipping chunk missing text content: {chunk}")
                continue
        else:
            logger.warning(f"Skipping non-dict chunk: {chunk}")
            continue
    
    if not texts:
        logger.warning("No valid text content found for clustering")
        return {
            'labels': [],
            'metrics': {'error': 'No valid text content'},
            'algorithm': algorithm,
            'params': {
                'n_clusters': n_clusters if algorithm == "kmeans" else None,
                'min_samples': min_samples if algorithm == "dbscan" else None,
                'eps': eps if algorithm == "dbscan" else None
            }
        }
    
    embeddings = embeddings_model.embed_documents(texts)
    
    # Cluster documents
    if algorithm == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(embeddings)
    elif algorithm == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    
    # Calculate metrics
    metrics = {}
    if len(set(labels)) > 1:  # Silhouette requires at least 2 clusters
        metrics['silhouette'] = silhouette_score(embeddings, labels)
    
    logger.info(f"Clustering complete. Found {len(set(labels))} clusters")
    
    return {
        'labels': labels.tolist(),
        'metrics': metrics,
        'algorithm': algorithm,
        'params': {
            'n_clusters': n_clusters if algorithm == "kmeans" else None,
            'min_samples': min_samples if algorithm == "dbscan" else None,
            'eps': eps if algorithm == "dbscan" else None
        }
    }

def batch_cluster(
    chunks_list: List[List[dict]],
    n_clusters: int = 5,
    algorithm: str = "kmeans",
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """Cluster multiple sets of documents in parallel.
    
    Args:
        chunks_list: List of document chunk lists to cluster
        n_clusters: Number of clusters (for KMeans)
        algorithm: "kmeans" or "dbscan"
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of clustering results (same format as cluster_documents)
    """
    from concurrent.futures import ThreadPoolExecutor
    
    results = []
    
    def cluster_single(chunks):
        try:
            return cluster_documents(chunks, n_clusters, algorithm)
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            return {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    logger.info(f"Starting batch clustering of {len(chunks_list)} document sets")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(cluster_single, chunks) for chunks in chunks_list]
        
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Unexpected clustering error: {str(e)}")
                results.append({
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
    
    logger.info("Batch clustering complete")
    return results

def generate_cluster_summary(chunks: List[dict], labels: List[int]) -> Dict[int, dict]:
    """Generate summaries for each cluster.
    
    Args:
        chunks: List of document chunks with text and metadata
        labels: Cluster assignments for each chunk
        
    Returns:
        Dictionary mapping cluster IDs to summary information including:
        - size: Number of chunks in cluster
        - top_keywords: Most frequent keywords
        - sample_text: Representative text sample
    """
    from collections import defaultdict
    
    clusters = defaultdict(list)
    for chunk, label in zip(chunks, labels):
        clusters[label].append(chunk)
    
    summaries = {}
    for label, cluster_chunks in clusters.items():
        # Get top keywords
        all_keywords = []
        for chunk in cluster_chunks:
            all_keywords.extend(chunk['metadata'].get('keywords', []))
        
        from collections import Counter
        top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(5)]
        
        # Get sample text (first 200 chars of first chunk)
        sample_text = cluster_chunks[0]['text'][:200] + '...' if len(cluster_chunks[0]['text']) > 200 else cluster_chunks[0]['text']
        
        summaries[label] = {
            'size': len(cluster_chunks),
            'top_keywords': top_keywords,
            'sample_text': sample_text
        }
    
    return summaries

def load_and_chunk(
    file_path: str, 
    chunk_size: int = 512, 
    chunk_overlap: int = 50,
    cluster_id: int = None
) -> List[dict]:
    """Load and chunk documents from various file formats.
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens
        cluster_id: Optional cluster ID to include in metadata
        
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
            metadata = {
                'source': file_path,  # Required by langchain for source tracking
                'file_path': file_path,  # Duplicate source for compatibility
                **chunk.metadata,
                **enrich_metadata(chunk.page_content, file_path, cluster_id=cluster_id)
            }
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

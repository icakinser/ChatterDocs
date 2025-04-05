import os
import json
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
from document_processor import batch_process
from rag_pipeline import get_rag_response
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG API"""
    model_path: str = "models/llama-2-7b-chat.Q4_K_M.gguf"
    db_path: str = "document_db.jsonl"
    max_workers: int = 4
    chunk_size: int = 1000
    chunk_overlap: int = 200
    n_clusters: int = 5
    clustering_algorithm: str = "kmeans"
    min_samples: int = 2
    eps: float = 0.5
    n_ctx: int = 4048  # LLM context window size
    temperature: float = 0.7  # Default generation temperature
    max_tokens: int = 1512  # Default max tokens per response
    backup_versions: bool = True  # Whether to maintain document versions

class RAGAPI:
    """Programmatic interface for the RAG system
    
    Provides synchronous and asynchronous methods for:
    - Document processing and indexing
    - Querying with retrieval-augmented generation
    - System monitoring and management
    
    Example:
        >>> api = RAGAPI()
        >>> api.process_documents(["doc.pdf"])
        >>> response = api.query("What is this document about?")
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the RAG API with optional configuration
        
        Args:
            config: RAGConfig instance with custom settings. 
                   If None, uses default configuration.
        """
        self.config = config or RAGConfig()
        self._version = "1.0.0"
        self._llm = None  # Will be initialized on first use
        
    def process_documents(
        self, 
        file_paths: List[Union[str, Path]],
        batch_size: Optional[int] = None
    ) -> dict:
        """
        Process and index documents in parallel batches
        
        Args:
            file_paths: List of paths to documents to process
            batch_size: Number of documents to process in parallel
            
        Returns:
            Dictionary containing processing results:
            - processed: Number of successfully processed documents
            - failed: Number of failed documents
            - total_chunks: Total chunks generated
            - errors: List of error messages for failed documents
        """
        from document_processor import batch_process
        
        # Convert paths to strings and validate
        str_paths = [str(p) for p in file_paths]
        missing = [p for p in str_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Documents not found: {', '.join(missing)}")
            
        workers = batch_size or self.config.max_workers
        logger.info(f"Starting batch processing of {len(str_paths)} documents with {workers} workers")
        
        try:
            result = batch_process(str_paths, 
                                 max_workers=workers,
                                 chunk_size=self.config.chunk_size,
                                 chunk_overlap=self.config.chunk_overlap)
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise RuntimeError("Document processing failed") from e
        
        logger.info(
            f"Batch processing complete. Success: {result['stats']['processed']}, "
            f"Failed: {result['stats']['failed']}, "
            f"Chunks: {result['stats']['chunks_generated']}"
        )
        
        # Save processed chunks to database
        from json_storage import DocumentDatabase
        db = DocumentDatabase(self.config.db_path)
        db.save(result['results'])
        logger.info(f"Saved {len(result['results'])} chunks to {self.config.db_path}")
        
        return {
            'processed': result['stats']['processed'],
            'failed': result['stats']['failed'],
            'total_chunks': result['stats']['chunks_generated'],
            'errors': [
                {'file': e['file_path'], 'error': e['error']}
                for e in result['errors']
            ]
        }
    
    def query(
        self, 
        question: str,
        temperature: float = None,
        max_tokens: int = None,
        include_sources: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Query the RAG system with enhanced options
        
        Args:
            question: The question to ask
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum length of response
            include_sources: Whether to include source documents
            
        Returns:
            Generated response (str or dict with sources if requested)
        """
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        try:
            return get_rag_response(
                question, 
                self.config.db_path,
                temperature=temperature,
                max_tokens=max_tokens,
                include_sources=include_sources
            )
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise RuntimeError("Failed to generate response") from e
    
    def get_status(self) -> dict:
        """
        Get detailed system status and health
        
        Returns:
            Dictionary with comprehensive status information including:
            - version: API version
            - documents_processed: Number of documents in index
            - model_loaded: Whether LLM is available
            - storage_size: Size of document storage (bytes)
            - last_processed: Timestamp of last document processing
            - system_info: Platform and Python version
            - config: Current configuration
            - versions: Available document versions if versioning enabled
        """
        from json_storage import DocumentDatabase
        
        status = {
            "version": self._version,
            "documents_processed": 0,
            "model_loaded": os.path.exists(self.config.model_path),
            "storage_size": 0,
            "last_processed": None,
            "system_info": {
                "platform": os.uname().sysname,
                "python": os.sys.version
            },
            "config": {
                "model_path": self.config.model_path,
                "db_path": self.config.db_path,
                "max_workers": self.config.max_workers,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "n_ctx": self.config.n_ctx,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "backup_versions": self.config.backup_versions
            }
        }
        
        if os.path.exists(self.config.db_path):
            status["documents_processed"] = sum(
                1 for _ in open(self.config.db_path, 'r', encoding='utf-8')
            )
            status["storage_size"] = os.path.getsize(self.config.db_path)
            
            # Get last modified time
            mtime = os.path.getmtime(self.config.db_path)
            status["last_processed"] = datetime.fromtimestamp(mtime).isoformat()
            
        return status

    async def aquery(self, question: str, **kwargs) -> str:
        """Async version of query method"""
        # Implementation would use async LLM calls
        return self.query(question, **kwargs)

    async def aprocess_documents(self, file_paths: List[str], **kwargs) -> dict:
        """Async version of process_documents"""
        # Implementation would use async processing
        return self.process_documents(file_paths, **kwargs)

    def cluster_documents(
        self,
        chunks: List[dict],
        n_clusters: Optional[int] = None,
        algorithm: Optional[str] = None,
        min_samples: Optional[int] = None,
        eps: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Cluster document chunks based on their content
        
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
        from document_processor import cluster_documents
        
        params = {
            'n_clusters': n_clusters or self.config.n_clusters,
            'algorithm': algorithm or self.config.clustering_algorithm,
            'min_samples': min_samples or self.config.min_samples,
            'eps': eps or self.config.eps
        }
        
        try:
            result = cluster_documents(chunks, **params)
            logger.info(f"Clustering complete. Found {len(set(result['labels']))} clusters")
            return result
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            raise RuntimeError("Document clustering failed") from e

    def batch_cluster(
        self,
        chunks_list: List[List[dict]],
        n_clusters: Optional[int] = None,
        algorithm: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Cluster multiple sets of documents in parallel
        
        Args:
            chunks_list: List of document chunk lists to cluster
            n_clusters: Number of clusters (for KMeans)
            algorithm: "kmeans" or "dbscan"
            batch_size: Number of parallel clustering operations
            
        Returns:
            List of clustering results (same format as cluster_documents)
        """
        from document_processor import batch_cluster
        
        workers = batch_size or self.config.max_workers
        params = {
            'n_clusters': n_clusters or self.config.n_clusters,
            'algorithm': algorithm or self.config.clustering_algorithm
        }
        
        logger.info(f"Starting batch clustering of {len(chunks_list)} document sets with {workers} workers")
        
        try:
            results = batch_cluster(
                chunks_list,
                max_workers=workers,
                **params
            )
            logger.info("Batch clustering complete")
            return results
        except Exception as e:
            logger.error(f"Batch clustering failed: {str(e)}")
            raise RuntimeError("Batch clustering failed") from e

    def get_cluster_summary(
        self,
        chunks: List[dict],
        labels: List[int]
    ) -> Dict[int, dict]:
        """
        Generate summaries for each cluster
        
        Args:
            chunks: List of document chunks with text and metadata
            labels: Cluster assignments for each chunk
            
        Returns:
            Dictionary mapping cluster IDs to summary information including:
            - size: Number of chunks in cluster
            - top_keywords: Most frequent keywords
            - sample_text: Representative text sample
        """
        from document_processor import generate_cluster_summary
        
        try:
            return generate_cluster_summary(chunks, labels)
        except Exception as e:
            logger.error(f"Failed to generate cluster summaries: {str(e)}")
            raise RuntimeError("Cluster summary generation failed") from e

    async def acluster_documents(self, chunks: List[dict], **kwargs) -> Dict[str, Any]:
        """Async version of cluster_documents"""
        return self.cluster_documents(chunks, **kwargs)

    async def abatch_cluster(self, chunks_list: List[List[dict]], **kwargs) -> List[Dict[str, Any]]:
        """Async version of batch_cluster"""
        return self.batch_cluster(chunks_list, **kwargs)

    def create_version(self, version_name: str) -> None:
        """
        Create a named version of the current document database
        
        Args:
            version_name: Unique name for this version
        """
        from json_storage import DocumentDatabase
        try:
            db = DocumentDatabase(self.config.db_path)
            db._create_version(version_name)
            logger.info(f"Created document version: {version_name}")
        except Exception as e:
            logger.error(f"Failed to create version: {str(e)}")
            raise RuntimeError("Version creation failed") from e

    def get_version(self, version_name: str) -> List[Dict]:
        """
        Load a specific version of the document database
        
        Args:
            version_name: Name of version to load
            
        Returns:
            List of documents in the specified version
        """
        from json_storage import DocumentDatabase
        try:
            db = DocumentDatabase(self.config.db_path)
            return db.get_version(version_name)
        except Exception as e:
            logger.error(f"Failed to load version {version_name}: {str(e)}")
            raise RuntimeError("Version load failed") from e

    def list_versions(self) -> List[str]:
        """
        List available document versions
        
        Returns:
            List of version names
        """
        from json_storage import DocumentDatabase
        try:
            db = DocumentDatabase(self.config.db_path)
            return db.list_versions()
        except Exception as e:
            logger.error(f"Failed to list versions: {str(e)}")
            raise RuntimeError("Version listing failed") from e

    async def acreate_version(self, version_name: str) -> None:
        """Async version of create_version"""
        return self.create_version(version_name)

    async def aget_version(self, version_name: str) -> List[Dict]:
        """Async version of get_version"""
        return self.get_version(version_name)

    async def alist_versions(self) -> List[str]:
        """Async version of list_versions"""
        return self.list_versions()

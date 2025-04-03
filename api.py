import os
from typing import List, Optional
from document_processor import process_document
from rag_pipeline import get_rag_response
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAPI:
    """Programmatic interface for the RAG system"""
    
    def __init__(self, model_path: str = "models/llama-2-7b-chat.Q4_K_M.gguf"):
        """
        Initialize the RAG API
        
        Args:
            model_path: Path to GGUF model file
        """
        self.model_path = model_path
        self.db_path = "document_db.jsonl"
        
    def process_documents(self, file_paths: List[str], batch_size: int = 4) -> dict:
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
        
        # Validate all files exist first
        missing = [p for p in file_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Documents not found: {', '.join(missing)}")
            
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        result = batch_process(file_paths, max_workers=batch_size)
        
        logger.info(
            f"Batch processing complete. Success: {result['stats']['processed']}, "
            f"Failed: {result['stats']['failed']}, "
            f"Chunks: {result['stats']['chunks_generated']}"
        )
        
        return {
            'processed': result['stats']['processed'],
            'failed': result['stats']['failed'],
            'total_chunks': result['stats']['chunks_generated'],
            'errors': [
                {'file': e['file_path'], 'error': e['error']}
                for e in result['errors']
            ]
        }
    
    def query(self, question: str) -> str:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            
        Returns:
            Generated response
        """
        return get_rag_response(question, self.db_path)
    
    def get_status(self) -> dict:
        """
        Get system status
        
        Returns:
            Dictionary with status information including:
            - documents_processed: Number of documents in index
            - model_loaded: Whether LLM is available
            - storage_size: Size of document storage
        """
        status = {
            "documents_processed": 0,
            "model_loaded": os.path.exists(self.model_path),
            "storage_size": 0
        }
        
        if os.path.exists(self.db_path):
            status["documents_processed"] = sum(
                1 for _ in open(self.db_path, 'r', encoding='utf-8')
            )
            status["storage_size"] = os.path.getsize(self.db_path)
            
        return status

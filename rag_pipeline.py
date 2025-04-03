import logging
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from json_storage import load_from_jsonl
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache for processed documents
_document_cache = {
    "last_modified": 0,
    "documents": None
}

def get_rag_response(query: str, db_path: str = "document_db.jsonl") -> str:
    """Process a user query using RAG pipeline.
    Uses cached documents unless the source file has changed.
    
    Args:
        query: User's question
        db_path: Path to JSONL file with document chunks
        
    Returns:
        Generated response
    """
    logger.info(f"Processing query: '{query}' using db at {db_path}")
    
    # Check cache and only reprocess if documents changed
    current_modified = os.path.getmtime(db_path)
    if (_document_cache["documents"] is None or 
        current_modified > _document_cache["last_modified"]):
        logger.info("Documents changed - reprocessing")
        dict_docs = load_from_jsonl(db_path)
        documents = [
            Document(
                page_content=doc["text"],
                metadata=doc["metadata"]
            ) for doc in dict_docs
        ]
        documents = filter_complex_metadata(documents)
        _document_cache["documents"] = documents
        _document_cache["last_modified"] = current_modified
        logger.info(f"Processed and cached {len(documents)} documents")
    else:
        documents = _document_cache["documents"]
        logger.info(f"Using cached {len(documents)} documents")
    
    # Create embeddings
    logger.info("Creating document embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store with persistence
    logger.info("Building vector store")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    db.persist()
    logger.info("Vector store created and persisted")
    
    # Load LLM
    logger.info("Initializing LLM")
    llm = LlamaCpp(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
        n_ctx=2048,
        verbose=True
    )
    logger.info("LLM initialized")
    
    # Create hybrid retriever combining vector and keyword search
    logger.info("Creating hybrid retriever")
    from langchain.retrievers import BM25Retriever, EnsembleRetriever
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    
    # Create keyword retriever
    texts = [doc.page_content for doc in documents]
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = 2
    
    # Create vector retriever
    vector_retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Combine retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    
    # Create QA chain with hybrid retriever
    logger.info("Creating QA chain with hybrid retrieval")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever
    )
    
    logger.info("Generating response...")
    response = qa_chain.run(query)
    logger.info(f"Successfully generated response for query: '{query}'")
    
    return response

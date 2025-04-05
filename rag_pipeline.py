import logging
from datetime import datetime
from typing import Union, Dict, Any
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

def get_rag_response(
    query: str, 
    db_path: str = "document_db.jsonl",
    temperature: float = 0.7,
    max_tokens: int = 512,
    include_sources: bool = False
) -> Union[str, Dict[str, Any]]:
    """Process a user query using RAG pipeline.
    Uses cached documents unless the source file has changed.
    
    Args:
        query: User's question
        db_path: Path to JSONL file with document chunks
        temperature: Controls randomness (0.0-1.0)
        max_tokens: Maximum length of response
        include_sources: Whether to include source documents
        
    Returns:
        Generated response (str or dict with sources if requested)
    """
    logger.info(f"Processing query: '{query}' using db at {db_path}")
    
    try:
        # Check cache and only reprocess if documents changed
        current_modified = os.path.getmtime(db_path)
        if (_document_cache["documents"] is None or 
            current_modified > _document_cache["last_modified"]):
            logger.info("Documents changed - reprocessing")
            raw_docs = load_from_jsonl(db_path)
            documents = []
            for doc in raw_docs:
                try:
                    if isinstance(doc, dict):
                        documents.append(Document(
                            page_content=doc.get("text", ""),
                            metadata=doc.get("metadata", {})
                        ))
                    else:
                        logger.warning("Skipping non-dict document: %s", type(doc))
                except Exception as e:
                    logger.error("Error processing document: %s", str(e))
            documents = filter_complex_metadata(documents)
            _document_cache["documents"] = documents
            _document_cache["last_modified"] = current_modified
            logger.info(f"Processed and cached {len(documents)} documents")
        else:
            documents = _document_cache["documents"]
            logger.info(f"Using cached {len(documents)} documents")
        
        if not documents:
            logger.warning("No valid documents found for processing")
            return "I don't have enough information to answer that question. Please try adding some documents first."

        # Create embeddings
        logger.info("Creating document embeddings")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector store
        logger.info("Building vector store")
        for doc in documents:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = doc.metadata.get('file_name', 'Unknown source')
                
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        db.persist()
        
        # Load LLM
        logger.info(f"Initializing LLM with temperature={temperature}, max_tokens={max_tokens}")
        llm = LlamaCpp(
            model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
            n_ctx=2048,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=True
        )
        
        # Create hybrid retriever
        logger.info("Creating hybrid retriever")
        from langchain.retrievers import BM25Retriever, EnsembleRetriever
        texts = [doc.page_content for doc in documents]
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = 2
        vector_retriever = db.as_retriever(search_kwargs={"k": 3})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        
        # Generate response with error handling
        logger.info("Generating response...")
        try:
            if include_sources:
                from langchain.chains import RetrievalQAWithSourcesChain
                qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=ensemble_retriever
                )
                result = qa_chain({"question": query})
                if not isinstance(result, dict) or 'answer' not in result:
                    raise ValueError("Invalid response format from QA chain")
                response = {
                    "answer": result["answer"],
                    "sources": result.get("sources", [])
                }
            else:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=ensemble_retriever
                )
                response = qa_chain.run(query)
                if not response:
                    raise ValueError("Empty response from QA chain")
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            fallback = qa_chain.run(query) if 'qa_chain' in locals() else "ERROR WITH GENERATING RESPONSE"
            return {
                "error": "ERROR WITH GENERATING RESPONSE",
                "answer": fallback
            }
        
        logger.info(f"Successfully generated response for query: '{query}'")
        return response

    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}", exc_info=True)
        return {
            "error": "ERROR WITH GENERATING RESPONSE",
            "answer": "ERROR WITH GENERATING RESPONSE"
        }

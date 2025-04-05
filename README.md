# ChatterDocs - AI Document Companion

A local Retrieval-Augmented Generation (RAG) system that allows you to chat with your documents without cloud dependencies.

## Features

- **Document Processing**: Supports PDF, TXT, RTF, HTML, and DJVU formats with parallel batch processing
- **Local Vector Storage**: Uses ChromaDB for efficient document indexing  
- **Local LLM**: Runs llama-2-7b-chat (4-bit quantized) locally with automatic GPU acceleration
- **Web Interface**: Simple Streamlit-based UI for document upload and chat
- **Persistent Storage**: Documents and embeddings are saved between sessions
- **Database Management**:
  - Versioned document storage with automatic versioning on upload
  - Manual backup creation
  - Version selection and loading
  - Automatic timestamped backups

## Requirements

- Python 3.8+
- Optional: For GPU acceleration, install PyTorch with Metal support (macOS) or CUDA (NVIDIA GPUs)
- Required Python packages (see requirements.txt for exact versions):
  ```
  streamlit
  chromadb
  llama-cpp-python
  nltk
  pypdf
  python-docx
  beautifulsoup4
  langchain-community
  langchain-core
  rake-nltk
  ```

## Installation

1. Clone this repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK data:
   ```bash
   python download_nltk.py
   ```
4. Place your GGUF model in the `models/` directory (llama-2-7b-chat.Q4_K_M.gguf is recommended)

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```
2. Upload documents:
   - Click "Browse files" in the sidebar
   - Supported formats: PDF, TXT, RTF, HTML, DJVU
   - Documents will be processed and stored in `uploaded_docs/`

3. Chat with your documents:
   - Enter questions in the chat interface
   - The system will retrieve relevant passages and generate answers
   - Example queries:
     - "Summarize the main points of this document"
     - "What are the key findings in chapter 3?"
     - "Explain the concept of DNA replication"

## Project Structure

```
├── app.py                # Main Streamlit application
├── document_processor.py # Document loading and chunking
├── rag_pipeline.py       # RAG query processing
├── json_storage.py       # Document metadata storage
├── llm_config.py         # LLM configuration
├── chroma_db/            # Vector database storage
├── models/               # LLM model storage
├── uploaded_docs/        # User-uploaded documents
├── test_docs/            # Test documents
└── test_output/          # Test output files
```

## Database Management

The system provides comprehensive database management through the `DocumentDatabase` class:

```python
from json_storage import DocumentDatabase

# Initialize database
db = DocumentDatabase()

# Save documents with version name
db.save(documents, "v1_initial_upload")

# List available versions
versions = db.list_versions()  # Returns ['v1_initial_upload', ...]

# Load specific version
documents = db.get_version("v1_initial_upload")

# Create manual backup (auto-timestamped)
db.save(documents)  # Creates backup at db_backups/backup_TIMESTAMP.jsonl

# Restore from backup
backup_files = sorted(glob("db_backups/*.jsonl"))
latest_backup = backup_files[-1]
documents = db.load(latest_backup)
```

## New Features (v1.1)

- **Enhanced Document Clustering**:
  - K-means and DBSCAN algorithms
  - Automatic cluster summarization
  - Batch clustering for large datasets

- **Improved API**:
  - Temperature and max_tokens control
  - Source document tracking
  - Detailed version management
  - Comprehensive system status

- **Robust Error Handling**:
  - Fallback mechanisms
  - Detailed error reporting
  - Processing retries

For complete API documentation including code examples, see [API_DOCS.md](API_DOCS.md)

## Programmatic Interface

For developers who want to integrate the RAG system into their applications, we provide a comprehensive Python API documented in [API_DOCS.md](API_DOCS.md). Key capabilities include:

```python
from api import RAGAPI

# Initialize the API
rag = RAGAPI()

# Process documents
rag.process_documents(["documents/sample.pdf"])

# Query the system
response = rag.query("What is the main topic of this document?")
print(response)

# Get system status
status = rag.get_status()
print(f"Documents processed: {status['documents_processed']}")
```

Key Methods:
- `process_documents(file_paths, batch_size=4)`: 
  - Index one or more documents in parallel batches
  - Returns processing statistics including:
    - processed: Number of successful documents
    - failed: Number of failed documents
    - total_chunks: Total text chunks generated
    - errors: List of error messages for failed documents
- `query(question)`: Get an answer to your question (documents are only reprocessed if the source file changes)
- `get_status()`: Check system status and document count

Note: The system caches processed documents and only reprocesses them when the source file (document_db.jsonl) is modified. This ensures efficient querying while maintaining data freshness.

## Future Improvements

Planned enhancements to the RAG pipeline:

1. **Advanced Chunking**:
   - Implement semantic-aware document splitting
   - Add table/chart extraction from documents
   - Support for document structure preservation

2. **Enhanced Retrieval**:
   - Hybrid search combining semantic, keyword, and metadata filters
   - Query expansion and rewriting
   - Reranking of retrieved passages

3. **LLM Optimization**:
   - Support for larger models (13B, 70B)
   - Prompt engineering templates
   - Response validation and fact-checking

4. **Performance**:
   - GPU acceleration for faster inference (implemented)
   - Batch processing for large document collections (implemented)
   - Background indexing of new documents (planned) 
   - Distributed processing support (planned)

5. **Monitoring**:
   - Query logging and analytics
   - Performance metrics tracking
   - Quality evaluation framework

## Known Limitations

- Processing large PDFs (>100 pages) may be slow
- HTML/DJVU parsing depends on document structure quality
- The 7B model may struggle with complex reasoning tasks
- First-time document processing requires significant RAM
- Batch processing performance depends on available CPU cores and document sizes

## Screenshots

(Add screenshots showing:)
- Document upload interface
- Sample query responses
- Database management sidebar with version selection
- Backup creation confirmation

## Troubleshooting

- If documents fail to process:
  - Check the file format is supported
  - Verify the document isn't password protected
  - Ensure sufficient disk space is available

- If the LLM fails to load:
  - Verify the model file is in `models/`
  - Check the filename matches `llama-2-7b-chat.Q4_K_M.gguf`

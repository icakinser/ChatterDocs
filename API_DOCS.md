# RAG API Documentation

## Overview
The RAG API provides programmatic access to document processing, clustering, and query capabilities. Key features include:

- Batch document processing with parallel workers
- Document versioning and backup management
- Automatic text chunking and metadata enrichment
- Document clustering by semantic similarity
- Retrieval-augmented generation with local LLM

## Initialization

```python
from api import RAGAPI, RAGConfig

# With default configuration
api = RAGAPI()

# With custom configuration
config = RAGConfig(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    db_path="custom_db.jsonl",
    max_workers=4,
    chunk_size=1000,
    chunk_overlap=200,
    n_clusters=5
)
api = RAGAPI(config)
```

## Document Processing

```python
# Process single document
result = api.process_documents(["doc.pdf"])

# Process multiple documents in parallel
result = api.process_documents(["doc1.pdf", "doc2.txt"], batch_size=2)

# Result structure:
{
    'processed': 2,
    'failed': 0,
    'total_chunks': 15,
    'errors': []
}
```

## Version Management

```python
# Create named version
api.create_version("v1_initial_load")

# List available versions
versions = api.list_versions()  # Returns ['v1_initial_load', ...]

# Load specific version
docs = api.get_version("v1_initial_load")
```

## Document Clustering

```python
# Cluster existing documents
clusters = api.cluster_documents(
    chunks,
    n_clusters=3,
    algorithm="kmeans"
)

# Result structure:
{
    'labels': [0, 1, 2, 0, ...],
    'metrics': {'silhouette': 0.75},
    'algorithm': 'kmeans',
    'params': {'n_clusters': 3}
}

# Get cluster summaries
summaries = api.get_cluster_summary(chunks, clusters['labels'])
```

## Querying Documents

```python
# Basic query
response = api.query("What is the main topic?")

# Query with sources
response = api.query(
    "What are the key points?",
    include_sources=True
)

# With generation parameters
response = api.query(
    "Summarize this document",
    temperature=0.3,
    max_tokens=1000
)

# Response with sources:
{
    'answer': 'The document discusses...',
    'sources': [
        {'file': 'doc.pdf', 'text': '...'},
        {'file': 'doc.pdf', 'text': '...'}
    ]
}
```

## System Status

```python
status = api.get_status()

# Returns:
{
    'version': '1.0.0',
    'documents_processed': 5,
    'model_loaded': True,
    'storage_size': 10240,
    'last_processed': '2025-04-04T21:00:00',
    'config': {
        'model_path': 'models/llama-2-7b-chat.Q4_K_M.gguf',
        'db_path': 'document_db.jsonl',
        'max_workers': 4,
        'chunk_size': 512,
        'chunk_overlap': 200
    }
}
```

## Advanced Examples

### Batch Processing with Error Handling
```python
files = ["doc1.pdf", "doc2.pdf", "corrupt.pdf"]
result = api.process_documents(files)

if result['failed'] > 0:
    print(f"Failed to process {result['failed']} files:")
    for error in result['errors']:
        print(f"- {error['file']}: {error['error']}")
```

### Custom Clustering Configuration
```python
clusters = api.cluster_documents(
    chunks,
    algorithm="dbscan",
    eps=0.5,
    min_samples=2
)
```

### Query with Fallback
```python
try:
    response = api.query(question, include_sources=True)
except Exception as e:
    print(f"Error getting sources: {e}")
    response = api.query(question)  # Fallback without sources
```

## Error Handling
The API raises these exceptions:
- `FileNotFoundError`: When input files don't exist
- `ValueError`: For invalid parameters
- `RuntimeError`: For processing/query failures

Handle them with try/except blocks as shown in examples.

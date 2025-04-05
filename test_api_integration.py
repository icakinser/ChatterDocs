import os
import time
import json
from api import RAGAPI, RAGConfig
from typing import List, Dict, Any

def main():
    # Initialize API with test configuration
    config = RAGConfig(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
        db_path="test_db.jsonl",
        max_workers=2,
        chunk_size=512,
        backup_versions=True
    )
    
    # Ensure clean test environment
    if os.path.exists(config.db_path):
        os.remove(config.db_path)
    
    # Create empty database file
    if os.path.exists(config.db_path):
        os.remove(config.db_path)
    
    # We'll let the process_documents method create and populate the file
    
    api = RAGAPI(config)

    # Get list of test documents
    test_docs = [
        "test_docs/01_solar_energy.txt",
        "test_docs/02_wind_power.txt", 
        "test_docs/03_hydroelectric_power.txt",
        "test_docs/04_geothermal_energy.txt"
    ]

    # Test 1: Document Processing
    print("=== Testing Document Processing ===")
    process_results = api.process_documents(test_docs)
    print(f"Processed {process_results['processed']} documents")
    print(f"Generated {process_results['total_chunks']} chunks")
    
    # Test 2: Versioning System
    print("\n=== Testing Versioning ===")
    api.create_version("initial_load")
    versions = api.list_versions()
    print(f"Available versions: {versions}")
    
    # Test 3: Clustering
    print("\n=== Testing Clustering ===")
    # Load processed chunks from the database
    from json_storage import DocumentDatabase
    db = DocumentDatabase(config.db_path)
    chunks = db.load(ensure_dict=True)
    
    cluster_results = api.cluster_documents(
        chunks,
        n_clusters=3,
        algorithm="kmeans"
    )
    print(f"Cluster labels: {cluster_results['labels']}")
    
    # Test 4: Cluster Summarization
    summaries = api.get_cluster_summary(chunks, cluster_results['labels'])
    for cluster_id, summary in summaries.items():
        print(f"\nCluster {cluster_id} Summary:")
        print(f"Size: {summary['size']}")
        print(f"Top Keywords: {', '.join(summary['top_keywords'])}")
        print(f"Sample Text: {summary['sample_text']}")
    
    # Test 5: Querying (limited to 2 queries with logging)
    print("\n=== Testing Querying ===")
    queries = [
        "What does Robert look like?",
        "Where does Robert live?"
    ]
    
    log_file = "/Users/robertkinser/Desktop/rag_test_log.txt"
    with open(log_file, "w") as f:
        for query in queries:
            f.write(f"\nQuery: {query}\n")
            try:
                response = api.query(query, include_sources=True)
                if isinstance(response, dict):
                    f.write(f"Answer: {response['answer']}\n")
                    f.write(f"Sources: {response['sources']}\n")
                else:
                    f.write(f"Answer: {response}\n")
            except Exception as e:
                f.write(f"Error: {str(e)}\n")
                response = api.query(query, include_sources=False)
                f.write(f"Fallback Answer: {response}\n")
    
    print(f"Query results logged to {log_file}")
    
    # Test 6: Status Check
    print("\n=== System Status ===")
    status = api.get_status()
    print(f"Documents processed: {status['documents_processed']}")
    print(f"Storage size: {status['storage_size']} bytes")
    print(f"Last processed: {status['last_processed']}")

if __name__ == "__main__":
    main()

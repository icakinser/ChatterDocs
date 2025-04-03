import json
import os
import logging
import shutil
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from document_processor import load_and_chunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentDatabase:
    """Manages document storage with versioning and backups."""
    
    def __init__(self, db_path: str = "document_db.jsonl"):
        self.db_path = db_path
        self.backup_dir = "db_backups"
        self.versions_dir = "db_versions"
        Path(self.backup_dir).mkdir(exist_ok=True)
        Path(self.versions_dir).mkdir(exist_ok=True)
        self._loaded_documents: List[Dict] = []
        self._is_loaded = False
    
    def save(self, documents: List[Dict], version_name: Optional[str] = None) -> None:
        """Save documents to database with optional versioning.
        
        Args:
            documents: Documents to save
            version_name: Optional version name/tag
        """
        self._write_to_jsonl(documents, self.db_path)
        self._create_backup()
        if version_name:
            self._create_version(version_name)
        self._loaded_documents = documents
        self._is_loaded = True
    
    def load(self) -> List[Dict]:
        """Load documents from database."""
        if not self._is_loaded:
            self._loaded_documents = self._load_from_jsonl(self.db_path)
            self._is_loaded = True
        return self._loaded_documents
    
    def get_version(self, version_name: str) -> List[Dict]:
        """Load a specific version of the database."""
        version_path = os.path.join(self.versions_dir, f"{version_name}.jsonl")
        return self._load_from_jsonl(version_path)
    
    def list_versions(self) -> List[str]:
        """List available database versions."""
        return [f.replace('.jsonl', '') for f in os.listdir(self.versions_dir) 
                if f.endswith('.jsonl')]
    
    def _write_to_jsonl(self, documents: List[Dict], output_path: str) -> None:
        """Internal method to write documents to JSONL file."""
        logger.info(f"Writing {len(documents)} documents to {output_path}")
        with open(output_path, 'w') as f:
            for i, doc in enumerate(documents, 1):
                f.write(json.dumps(doc) + '\n')
                if i % 10 == 0 or i == len(documents):
                    logger.debug(f"Written {i}/{len(documents)} documents")
        logger.info(f"Successfully wrote {len(documents)} documents to {output_path}")
    
    def _load_from_jsonl(self, input_path: str) -> List[Dict]:
        """Internal method to load documents from JSONL file."""
        logger.info(f"Loading documents from {input_path}")
        documents = []
        with open(input_path, 'r') as f:
            for i, line in enumerate(f, 1):
                try:
                    documents.append(json.loads(line))
                    if i % 10 == 0 or i == len(documents):
                        logger.debug(f"Loaded {i} documents")
                except json.JSONDecodeError as e:
                    logger.error(f"Error loading line {i}: {str(e)}")
                    continue
        logger.info(f"Successfully loaded {len(documents)} documents from {input_path}")
        return documents
    
    def _create_backup(self) -> None:
        """Create timestamped backup of current database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}.jsonl")
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
    
    def _create_version(self, version_name: str) -> None:
        """Create a named version of the database."""
        version_path = os.path.join(self.versions_dir, f"{version_name}.jsonl")
        shutil.copy2(self.db_path, version_path)
        logger.info(f"Created version '{version_name}' at {version_path}")

def write_to_jsonl(documents: List[Dict], output_path: str) -> None:
    """Legacy function - use DocumentDatabase class instead."""
    logger.info(f"Writing {len(documents)} documents to {output_path}")
    with open(output_path, 'w') as f:
        for i, doc in enumerate(documents, 1):
            f.write(json.dumps(doc) + '\n')
            if i % 10 == 0 or i == len(documents):
                logger.debug(f"Written {i}/{len(documents)} documents")
    logger.info(f"Successfully wrote {len(documents)} documents to {output_path}")

def load_from_jsonl(input_path: str) -> List[Dict]:
    """Legacy function - use DocumentDatabase class instead."""
    logger.info(f"Loading documents from {input_path}")
    documents = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                documents.append(json.loads(line))
                if i % 10 == 0 or i == len(documents):
                    logger.debug(f"Loaded {i} documents")
            except json.JSONDecodeError as e:
                logger.error(f"Error loading line {i}: {str(e)}")
                continue
    logger.info(f"Successfully loaded {len(documents)} documents from {input_path}")
    return documents

def test_jsonl_storage():
    """Test JSONL storage functionality."""
    os.makedirs("test_output", exist_ok=True)
    
    # Process and store sample document
    chunks = load_and_chunk("test_docs/sample.txt")
    write_to_jsonl(chunks, "test_output/sample.jsonl")
    
    # Load and verify stored data
    loaded_chunks = load_from_jsonl("test_output/sample.jsonl")
    print("\nJSONL storage test results:")
    print(f"Saved {len(chunks)} chunks, loaded {len(loaded_chunks)} chunks")
    print("First loaded chunk:")
    for k, v in loaded_chunks[0].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    test_jsonl_storage()

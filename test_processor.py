from document_processor import load_and_chunk
import os

# Create test directory if it doesn't exist
os.makedirs("test_docs", exist_ok=True)

# Create more comprehensive sample test files
sample_text = """Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and human language. 
It focuses on how to program computers to process and analyze large amounts of natural language data.
Key tasks include text classification, named entity recognition, and sentiment analysis."""

with open("test_docs/sample.txt", "w") as f:
    f.write(sample_text)

# Test the processor
print("Testing text file processing with metadata enrichment:")
chunks = load_and_chunk("test_docs/sample.txt")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print("Text:", chunk["text"])
    print("Metadata:")
    for k, v in chunk["metadata"].items():
        print(f"  {k}: {v}")

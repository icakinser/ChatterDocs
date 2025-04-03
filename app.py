import streamlit as st
from document_processor import load_and_chunk
from json_storage import DocumentDatabase
from rag_pipeline import get_rag_response
import os

# Initialize document database
db = DocumentDatabase()

# App title
st.title("ChatterDocs - AI Document Companion")

# Sidebar for document upload
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, DOCX, HTML)",
        type=["pdf", "txt", "docx", "html"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        os.makedirs("uploaded_docs", exist_ok=True)
        for file in uploaded_files:
            file_path = os.path.join("uploaded_docs", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Process and store documents
            chunks = load_and_chunk(file_path)
            version_name = f"v_{len(db.list_versions()) + 1}_{file.name}"
            db.save(chunks, version_name)
            st.success(f"Processed {file.name} and saved as version '{version_name}'")

# Database management section
with st.sidebar:
    st.header("Database Management")
    
    if st.button("Create Backup"):
        db.save(db.load())
        st.success("Created new backup")
    
    versions = db.list_versions()
    if versions:
        selected_version = st.selectbox(
            "Load Version",
            versions,
            index=len(versions)-1
        )
        if st.button("Load Selected Version"):
            documents = db.get_version(selected_version)
            db.save(documents)
            st.success(f"Loaded version '{selected_version}'")

# Main chat interface
st.header("Chat with your documents")
user_query = st.text_input("Enter your question:")

if user_query:
    with st.spinner("Searching documents..."):
        response = get_rag_response(user_query)
        st.write(response)

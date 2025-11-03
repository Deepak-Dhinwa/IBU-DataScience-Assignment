"""
Document and text ingestion functions for the Personal Cognitive Memory Agent.
Handles PDF parsing, text processing, and vector database storage.
"""
import os
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import CHROMA_DB_PATH, CHROMA_NEW_VERSION, HAS_UNSTRUCTURED
from .models import embedding_model

# Import Chroma based on version
if CHROMA_NEW_VERSION:
    from langchain_chroma import Chroma
else:
    from langchain_community.vectorstores import Chroma


def ingest_document(file_path: str, folder_name: str) -> str:
    """
    Enhanced document ingestion with better PDF parsing for complex documents (PPT-converted PDFs, etc.)
    """
    try:
        print(f"Starting ingestion for {file_path} into folder '{folder_name}'...")
        
        if file_path.endswith(".pdf"):
            documents = _parse_pdf(file_path)
            if not documents:
                return f"Error: PDF file appears to be empty or could not be parsed: {file_path}"
                
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
            documents = loader.load()
        else:
            return f"Error: Unsupported file type: {file_path}"
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata
        current_time = int(time.time())
        for chunk in chunks:
            chunk.metadata["folder"] = folder_name
            chunk.metadata["source"] = file_path
            chunk.metadata["timestamp"] = current_time
            
        # Store in vector database
        if embedding_model is None:
            return "Error: Embedding model is not loaded."
            
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
        vector_store.add_documents(chunks)
        
        # Chroma 0.4.x+ auto-persists, so .persist() is no longer needed
        if not CHROMA_NEW_VERSION:
            try:
                vector_store.persist()
            except Exception:
                pass  # Already auto-persisted
                
        print(f"Successfully ingested {len(chunks)} chunks from {file_path}.")
        return f"Successfully ingested {file_path} into folder '{folder_name}'."
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return f"Error ingesting file: {e}"


def _parse_pdf(file_path: str) -> list:
    """
    Parse PDF using multiple methods with fallback.
    Tries PyMuPDF -> UnstructuredPDFLoader -> PyPDFLoader
    """
    documents = None
    load_error = None
    
    # Option 1: PyMuPDF (fitz) - Best for complex layouts and PPT-converted PDFs
    try:
        import fitz  # PyMuPDF
        print("Attempting PDF parsing with PyMuPDF (better for complex layouts)...")
        pdf_doc = fitz.open(file_path)
        pages = []
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            # Better text extraction with layout preservation
            text = page.get_text("text")
            if text.strip():  # Only add non-empty pages
                pages.append({"page_content": text, "metadata": {"page": page_num + 1}})
        pdf_doc.close()
        if pages:
            documents = [Document(page_content=p["page_content"], metadata=p["metadata"]) for p in pages]
            print(f"Successfully parsed PDF with PyMuPDF ({len(documents)} pages).")
    except ImportError:
        # PyMuPDF not installed, skip this method
        pass
    except Exception as e:
        load_error = f"PyMuPDF failed: {e}. "
        print(f"PyMuPDF parsing failed: {e}, trying fallback...")
    
    # Option 2: UnstructuredPDFLoader - Excellent for complex documents
    if documents is None and HAS_UNSTRUCTURED:
        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
            print("Attempting PDF parsing with UnstructuredPDFLoader (best for complex docs)...")
            loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="fast")
            documents = loader.load()
            if documents:
                print(f"Successfully parsed PDF with UnstructuredPDFLoader ({len(documents)} elements).")
        except Exception as e:
            load_error = (load_error or "") + f"UnstructuredPDFLoader failed: {e}. "
            print(f"UnstructuredPDFLoader parsing failed: {e}, trying fallback...")
    
    # Option 3: Fallback to PyPDFLoader (basic, but reliable)
    if documents is None:
        try:
            print("Using PyPDFLoader as fallback (basic PDF parsing)...")
            print(f"Note: {load_error or ''}PyPDFLoader may struggle with complex layouts from PPT-converted PDFs.")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if documents:
                print(f"Successfully parsed PDF with PyPDFLoader ({len(documents)} pages).")
        except Exception as e:
            error_msg = f"Error: All PDF parsing methods failed. {load_error or ''}Final error: {e}"
            print(error_msg)
            return None
    
    return documents


def ingest_text_note(text_content: str, folder_name: str) -> str:
    """
    Ingest a text note directly into the vector database.
    """
    try:
        print(f"Ingesting text note into folder '{folder_name}'...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.create_documents([text_content])
        
        current_time = int(time.time())
        for chunk in chunks:
            chunk.metadata["folder"] = folder_name
            chunk.metadata["source"] = "text_note"
            chunk.metadata["timestamp"] = current_time
            
        if embedding_model is None:
            return "Error: Embedding model is not loaded."
            
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
        vector_store.add_documents(chunks)
        
        # Chroma 0.4.x+ auto-persists, so .persist() is no longer needed
        if not CHROMA_NEW_VERSION:
            try:
                vector_store.persist()
            except Exception:
                pass  # Already auto-persisted
                
        print(f"Successfully ingested text note into folder '{folder_name}'.")
        return f"Successfully ingested text note into folder '{folder_name}'."
    except Exception as e:
        print(f"Error ingesting text note: {e}")
        return f"Error ingesting text note: {e}"


def list_ingested_documents() -> dict:
    """
    Lists all ingested documents with their metadata.
    Returns a dictionary with folder statistics and document details.
    """
    try:
        if embedding_model is None:
            return {"error": "Embedding model is not loaded."}
        
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
        
        # Get all documents with their metadata
        all_data = vector_store.get(include=["metadatas", "documents"])
        
        if not all_data or not all_data.get('metadatas'):
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "folders": {},
                "documents": []
            }
        
        metadatas = all_data['metadatas']
        documents = all_data.get('documents', [])
        
        # Group by source (unique documents)
        doc_info = {}
        folder_stats = {}
        
        for i, metadata in enumerate(metadatas):
            source = metadata.get('source', 'unknown')
            folder = metadata.get('folder', 'unknown')
            timestamp = metadata.get('timestamp', 0)
            
            # Track folder statistics
            if folder not in folder_stats:
                folder_stats[folder] = {"chunks": 0, "documents": set()}
            folder_stats[folder]["chunks"] += 1
            folder_stats[folder]["documents"].add(source)
            
            # Track unique documents
            if source not in doc_info:
                from datetime import datetime
                doc_info[source] = {
                    "source": source,
                    "folder": folder,
                    "chunks": 0,
                    "timestamp": timestamp,
                    "ingested_date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S") if timestamp else "Unknown"
                }
            doc_info[source]["chunks"] += 1
        
        # Convert folder stats to regular dict
        folder_stats_final = {}
        for folder, stats in folder_stats.items():
            folder_stats_final[folder] = {
                "chunks": stats["chunks"],
                "unique_documents": len(stats["documents"])
            }
        
        # Convert doc_info to list
        documents_list = list(doc_info.values())
        documents_list.sort(key=lambda x: x["timestamp"], reverse=True)  # Most recent first
        
        return {
            "total_chunks": len(metadatas),
            "total_documents": len(doc_info),
            "folders": folder_stats_final,
            "documents": documents_list
        }
    except Exception as e:
        print(f"Error listing documents: {e}")
        return {"error": f"Error listing documents: {e}"}


def delete_document(source_path: str) -> str:
    """
    Delete all chunks of a document by its source path.
    
    Args:
        source_path: The source path of the document to delete (as stored in metadata)
    
    Returns:
        Success or error message string
    """
    try:
        if embedding_model is None:
            return "Error: Embedding model is not loaded."
        
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
        
        # Get all chunks with this source to count them
        # Use the collection's get method directly
        all_data = vector_store._collection.get(
            where={"source": source_path},
            include=["metadatas"]
        )
        
        if not all_data or not all_data.get('ids'):
            return f"No documents found with source: {source_path}"
        
        num_chunks = len(all_data['ids'])
        
        # Delete all chunks with this source
        # ChromaDB supports deleting by filter using where parameter
        vector_store._collection.delete(where={"source": source_path})
        
        # Chroma 0.4.x+ auto-persists, so .persist() is no longer needed
        if not CHROMA_NEW_VERSION:
            try:
                vector_store.persist()
            except Exception:
                pass  # Already auto-persisted
        
        print(f"Successfully deleted {num_chunks} chunks from source: {source_path}")
        return f"Successfully deleted document '{os.path.basename(source_path)}' ({num_chunks} chunks removed)."
    except Exception as e:
        print(f"Error deleting document: {e}")
        return f"Error deleting document: {e}"


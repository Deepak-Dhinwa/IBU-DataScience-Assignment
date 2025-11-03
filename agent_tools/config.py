"""
Configuration and constants for the Personal Cognitive Memory Agent system.
"""
import os

# --- FEATURE FLAG ---
# This is the "switch" for your fine-tuned model.
# Set to False to use the llama3 fallback (for testing/demo).
# Set to True to attempt to load your fine-tuned Phi-3 model (requires high VRAM).
USE_CUSTOM_FINETUNED_MODEL = False
CUSTOM_MODEL_NAME = "insight-agent:latest"  # The name we will use after 'ollama create'

# --- EMBEDDING MODEL CONFIGURATION ---
# Options (in order of preference):
# 1. nomic-embed-text:latest - Good balance, 8192 context length, performs well
# 2. nomic-embed-text-v1.5 - Slightly better if available
# 3. mxbai-embed-large - Alternative high-performance model
EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"  # Change this to try different models

# --- DATABASE PATHS ---
CHROMA_DB_PATH = "./chroma_db"
ADAPTER_PATH = "./insight_writer_adapters"

# --- CHROMA VERSION HANDLING ---
# Check if new version is available (without importing it here)
try:
    import langchain_chroma
    CHROMA_NEW_VERSION = True
except ImportError:
    CHROMA_NEW_VERSION = False

# --- UNSTRUCTURED PDF SUPPORT ---
try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False


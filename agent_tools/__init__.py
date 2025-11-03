"""
Personal Cognitive Memory Agent - Refactored Modular Version

This module provides backward compatibility with the original agent_tools.py
by re-exporting all functions and variables.

The codebase has been refactored into:
- config.py: Configuration and constants
- models.py: Model loading and initialization
- ingestion.py: Document and text ingestion
- retrieval.py: Query retrieval and refinement
- agents/: Specialized analysis agents
  - insight_writer.py
  - pattern_miner.py
  - global_miner.py
  - visualization.py
"""

# Import all public functions and variables for backward compatibility
from .config import (
    USE_CUSTOM_FINETUNED_MODEL,
    CUSTOM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_PATH,
    ADAPTER_PATH,
    CHROMA_NEW_VERSION,
    HAS_UNSTRUCTURED
)

from .models import (
    llm,
    embedding_model,
    insight_writer_pipeline,
    pattern_miner_pipeline,
    initialize_all_models,
    load_insight_writer_pipeline,
    set_local_insight_writer_dir,
    set_local_pattern_miner_dir,
    set_use_ollama_default
)

from .ingestion import (
    ingest_document,
    ingest_text_note,
    list_ingested_documents,
    delete_document
)

from .retrieval import (
    recall_from_folder
)

from .agents import (
    run_insight_writer,
    run_pattern_miner,
    run_global_stratified_miner,
    run_viz_agent
)

# Export everything for backward compatibility
__all__ = [
    # Config
    'USE_CUSTOM_FINETUNED_MODEL',
    'CUSTOM_MODEL_NAME',
    'EMBEDDING_MODEL_NAME',
    'CHROMA_DB_PATH',
    'ADAPTER_PATH',
    'CHROMA_NEW_VERSION',
    'HAS_UNSTRUCTURED',
    # Models
    'llm',
    'embedding_model',
    'insight_writer_pipeline',
    'pattern_miner_pipeline',
    'initialize_all_models',
    'load_insight_writer_pipeline',
    'set_local_insight_writer_dir',
    'set_local_pattern_miner_dir',
    'set_use_ollama_default',
    # Ingestion
    'ingest_document',
    'ingest_text_note',
    'list_ingested_documents',
    'delete_document',
    # Retrieval
    'recall_from_folder',
    # Agents
    'run_insight_writer',
    'run_pattern_miner',
    'run_global_stratified_miner',
    'run_viz_agent'
]


"""
Global Stratified Miner Agent - Map-Reduce analysis across multiple folders.
Finds cross-folder patterns and generates important summaries.
"""
from datetime import datetime, date
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import CHROMA_DB_PATH, CHROMA_NEW_VERSION
from ..models import llm, embedding_model

# Import Chroma based on version
if CHROMA_NEW_VERSION:
    from langchain_chroma import Chroma
else:
    from langchain_community.vectorstores import Chroma


def run_global_stratified_miner(start_date: Optional[date] = None, end_date: Optional[date] = None, folders: Optional[list] = None) -> str:
    """
    Enhanced Global Stratified Miner with improved sampling strategy and multi-step Map-Reduce analysis.
    
    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
        folders: Optional list of folders to analyze. If None, analyzes all folders.
    """
    try:
        print("Running Enhanced Global Stratified Miner...")
        if llm is None:
            return "Error: Base LLM is not loaded."
        if embedding_model is None:
            return "Error: Embedding model is not loaded."
        
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
        all_metadatas = vector_store.get(include=["metadatas"])['metadatas']
        if not all_metadatas:
            return "The database is empty. Please ingest some documents first."
        
        # Get folders to analyze
        unique_folders_all = list(set([m['folder'] for m in all_metadatas if 'folder' in m]))
        if not unique_folders_all:
            return "No folders found in the database to analyze."
        
        # If specific folders provided, use those; otherwise use all
        folders_to_analyze = folders if folders else unique_folders_all
        folders_to_analyze = [f for f in folders_to_analyze if f in unique_folders_all]
        
        if not folders_to_analyze:
            return f"No valid folders found. Available folders: {unique_folders_all}"
        
        print(f"Analyzing folders: {folders_to_analyze}")

        # Build date filter (Chroma needs separate $gte and $lte conditions)
        date_filter = _build_date_filter(start_date, end_date)

        # STEP 1: MAP PHASE - Analyze each folder independently with diverse sampling
        folder_analyses = _map_phase(vector_store, folders_to_analyze, date_filter)
        
        if not folder_analyses:
            return "No data found in the specified date range for the selected folders."

        # STEP 2: REDUCE PHASE - Find cross-folder patterns
        response = _reduce_phase(folder_analyses, folders_to_analyze, start_date, end_date)
        
        # Post-process: Trim if too verbose
        if len(response) > 350:
            print("Global Miner: Response too verbose, trimming...")
            response = _trim_response(response)
        
        return response
    except Exception as e:
        print(f"Error in Global Miner: {e}")
        return f"Error in Global Miner: {e}"


def _build_date_filter(start_date: Optional[date], end_date: Optional[date]):
    """Build date filter for ChromaDB."""
    if start_date and end_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        start_ts = int(start_datetime.timestamp())
        end_ts = int(end_datetime.timestamp())
        # Chroma requires separate conditions, not combined in one dict
        return {"$and": [
            {"timestamp": {"$gte": start_ts}},
            {"timestamp": {"$lte": end_ts}}
        ]}
    return {}


def _map_phase(vector_store, folders_to_analyze, date_filter):
    """Map phase: Analyze each folder independently."""
    folder_analyses = []
    sampling_queries = [
        "recent activities and events",
        "important information and details",
        "patterns and trends",
        "key decisions and outcomes",
        "challenges and solutions"
    ]
    
    for folder in folders_to_analyze:
        print(f"  [MAP] Analyzing folder: {folder}")
        folder_filter = {"folder": folder}
        if date_filter:
            folder_filter = {"$and": [{"folder": folder}, date_filter]}
        
        # Sample documents from folder
        folder_docs = []
        for query in sampling_queries:
            try:
                docs = vector_store.similarity_search(
                    query=query,
                    k=3,  # 3 per query = 15 total per folder
                    filter=folder_filter if folder_filter else None
                )
                folder_docs.extend(docs)
            except Exception as e:
                print(f"    Warning: Error with query '{query}': {e}")
        
        # Remove duplicates
        seen_content = set()
        unique_docs = []
        for doc in folder_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        if unique_docs:
            folder_context = "\n\n".join([doc.page_content for doc in unique_docs[:20]])
            
            # Analyze this folder's data
            folder_analysis = _analyze_folder(folder, folder_context)
            folder_analyses.append(folder_analysis)
    
    return folder_analyses


def _analyze_folder(folder: str, folder_context: str) -> str:
    """Analyze a single folder's data."""
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze folder data for patterns. Categorize into Statistical, Quantitative, Qualitative, Relationships.

**Extract:**
- **Statistical**: Trends, frequencies, counts
- **Quantitative**: Numbers, percentages, ratios
- **Qualitative**: Themes, topics, sentiment
- **Relationships**: Connections between concepts

**Output (CONCISE - max 150 words):**
```
=== [Folder Name] ===

**Statistical:** [Trends/frequencies if any, else "None"]
- [Pattern: evidence]

**Quantitative:** [Metrics if any, else "None"]
- [Metric: value]

**Qualitative:** [Themes if any, else "None"]
- [Theme: evidence]

**Relationships:** [Connections if any, else "None"]
- [Relationship: evidence]

Summary: [2 sentences - key findings]
```

**Rules:**
- Use ONLY explicit data
- Be specific with evidence
- Cut fluff

**Folder:** {folder}
**Data:**
{data}"""),
        ("user", "Analyze this folder for categorized patterns (max 150 words). Provide evidence.")
    ])
    
    map_chain = map_prompt | llm | StrOutputParser()
    return map_chain.invoke({"folder": folder, "data": folder_context})


def _reduce_phase(folder_analyses, folders_to_analyze, start_date, end_date):
    """Reduce phase: Find cross-folder patterns."""
    combined_analysis = "\n\n".join([f"\n{folder_analysis}" for folder_analysis in folder_analyses])
    
    print(f"Running 'REDUCE' step: Finding cross-folder patterns across {len(folder_analyses)} folders...")
    
    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", """Find cross-folder patterns and create important summary. Categorize patterns like Pattern Miner.

**Categorize Cross-Folder Patterns:**
1. **Statistical**: Trends/frequencies across folders
2. **Quantitative**: Metrics/numbers spanning folders
3. **Qualitative**: Themes common to multiple folders
4. **Relationships**: Connections between folder contents

**Output Format:**
```
=== GLOBAL STRATIFIED ANALYSIS ===

Folders: {folders}
Date Range: {date_range}

**Cross-Folder Patterns:**

**Statistical Patterns:**
- [Pattern] (Folders: [X, Y]) - [evidence]

**Quantitative Patterns:**
- [Pattern] (Folders: [X, Y]) - [evidence]

**Qualitative Patterns:**
- [Pattern] (Folders: [X, Y]) - [evidence]

**Relationships:**
- [Pattern] (Folders: [X, Y]) - [evidence]

**Important Summary (Key Insights Only):**
[3-5 bullet points - ONLY the most important findings, not everything]
- [Most significant finding 1]
- [Most significant finding 2]
- [Most significant finding 3]

**Recommendation:**
[1 sentence if action needed, else "None"]
```

**RULES:**
- If folders are unrelated, say "Folders contain independent information. No cross-folder patterns."
- Summary should be 3-5 bullet points max - only most important
- Evidence must be from folder analyses
- NO generic patterns

**Folder Analyses:**
{analyses}"""),
        ("user", "Find cross-folder patterns (categorized) and create important summary (3-5 bullets max). Max 250 words total.")
    ])
    
    # Format date range for display
    date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}" if start_date and end_date else "All dates"
    
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    return reduce_chain.invoke({
        "folders": ", ".join(folders_to_analyze),
        "date_range": date_range_str,
        "analyses": combined_analysis
    })


def _trim_response(response: str) -> str:
    """Trim verbose responses to max 250 words."""
    trim_prompt = ChatPromptTemplate.from_messages([
        ("system", """Trim this analysis to max 250 words. Remove fluff, keep only real patterns with evidence and important summary (3-5 bullets)."""),
        ("user", "Trim to max 250 words:\n\n{analysis}")
    ])
    trim_chain = trim_prompt | llm | StrOutputParser()
    return trim_chain.invoke({"analysis": response})


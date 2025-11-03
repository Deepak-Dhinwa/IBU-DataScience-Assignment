"""
Retrieval and reasoning layer for querying the vector database.
Handles multi-folder queries, date filtering, and response generation.
"""
import os
from datetime import datetime, date
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import CHROMA_DB_PATH, CHROMA_NEW_VERSION
from .models import llm, embedding_model

# Import Chroma based on version
if CHROMA_NEW_VERSION:
    from langchain_chroma import Chroma
else:
    from langchain_community.vectorstores import Chroma

def recall_from_folder(query: str, folders: list, top_k: int, start_date: Optional[date] = None, end_date: Optional[date] = None) -> tuple:
    """
    Retrieve and synthesize information from specified folders with optional date filtering.
    
    Returns:
        (answer, raw_context, citations)
    """
    try:
        print(f"Recalling from folders {folders} with query: '{query}'")
        if embedding_model is None:
            return "Error: Embedding model is not loaded.", "", []
        if llm is None:
            return "Error: Base LLM is not loaded.", "", []
            
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
        
        # Build filter metadata
        filter_metadata = _build_chroma_filter(folders, start_date, end_date)
        print(f"Using filter: {filter_metadata}")
        
        # Retrieve documents
        search_kwargs = {"k": top_k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata
            
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        retrieved_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        citations = list(set([
            f"Source: {os.path.basename(doc.metadata.get('source', 'unknown'))} (Folder: {doc.metadata.get('folder', 'unknown')})"
            for doc in retrieved_docs
        ]))
        
        if not context:
            return "I found no relevant information for your query with the selected filters.", "", []

        # Generate refined answer using "Refiner Agent"
        answer = _refine_answer(query, context)
        return answer, context, citations
    except Exception as e:
        print(f"Error during recall: {e}")
        return f"Error recalling from folder: {e}", "", []


def _build_chroma_filter(folders: list, start_date: Optional[date] = None, end_date: Optional[date] = None):
    """
    Build ChromaDB filter metadata from folders and date range.
    Handles the bug fix for Chroma's filter requirements.
    """
    filter_parts = []
    
    if folders:
        # Create a list of {"folder": folder} dicts
        folder_or_clauses = [{"folder": f} for f in folders]
        # Use "$or" if there are multiple folders
        if len(folder_or_clauses) > 1:
            filter_parts.append({"$or": folder_or_clauses})
        else:
            filter_parts.append(folder_or_clauses[0])
    
    if start_date and end_date:
        # BUG FIX: Chroma doesn't support $gte and $lte in same dict
        # Need to split into separate $and conditions
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        start_ts = int(start_datetime.timestamp())
        end_ts = int(end_datetime.timestamp())
        filter_parts.append({"timestamp": {"$gte": start_ts}})
        filter_parts.append({"timestamp": {"$lte": end_ts}})
    
    # Combine all filter parts using "$and"
    # If no filters, use None (searches all documents)
    if len(filter_parts) > 1:
        return {"$and": filter_parts}
    elif len(filter_parts) == 1:
        return filter_parts[0]
    else:
        return None


def _refine_answer(query: str, context: str) -> str:
    """
    Use the "Refiner Agent" to synthesize retrieved chunks into a clear answer.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a factual refining agent that synthesizes messy data chunks into clear, accurate answers.

**Your Task:**
Synthesize the provided data chunks into a clean, direct, factual answer to the user's query.

**Rules:**
1. Answer ONLY based on the provided context - do not use outside knowledge
2. Do NOT add opinions, reflections, or personal insights
3. Combine related information from multiple chunks logically
4. If information is missing or unclear, state that explicitly
5. Use clear, structured formatting (bullets or numbered lists if helpful)
6. Cite which parts of the context support your answer when relevant

**When the answer is not in the context:**
Clearly state: "Based on the available data, I cannot find information to answer this query. The context does not contain relevant information about [specific aspect of the query]."

**Context Data:**
{context}"""),
        ("user", "User Query: {query}\n\nProvide a clear, factual answer based only on the context above.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "query": query})


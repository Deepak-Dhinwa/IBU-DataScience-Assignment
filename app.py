import streamlit as st
import agent_tools
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import os

# --- NEW IMPORT ---
from langchain_ollama import ChatOllama

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Personal Cognitive Memory Agent",
    layout="wide",
)

# --- 2. LOAD THE LLM (OLLAMA) ---
# We load the *fallback* llm here for the Query Enhancer.
# agent_tools.llm will be our primary model (which might be llama3 or custom)
try:
    llm_enhancer = ChatOllama(model="llama3:latest")
    print("ChatOllama (llama3:latest) loaded successfully for app.py's Query Enhancer.")
except Exception as e:
    st.error(f"Could not load ChatOllama. Is 'ollama serve' running? Error: {e}")
    st.stop()

# --- 3. HELPER FUNCTION & SESSION STATE ---

def new_query_session():
    """
    Clears the session state for the main chat/query context.
    This is our "New Query" button.
    """
    st.session_state.messages = []
    st.session_state.raw_context = None
    st.session_state.citations = []
    st.session_state.viz_charts = []
    st.session_state.viz_description = ""
    print("New query session started. Context cleared.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "raw_context" not in st.session_state:
    st.session_state.raw_context = None
if "citations" not in st.session_state:
    st.session_state.citations = []
if "viz_charts" not in st.session_state:
    st.session_state.viz_charts = []
if "viz_description" not in st.session_state:
    st.session_state.viz_description = ""

# --- 4. SIDEBAR (Ingestion & Global Controls) ---
st.sidebar.title("Personal Memory")
st.sidebar.markdown("This is the 'Ingestion Layer' of agent.")

# --- Ingestion ---
with st.sidebar.expander("Ingest New Memory", expanded=True):
    uploaded_file = st.file_uploader("Upload a document (.pdf, .txt)", type=["pdf", "txt"])
    ingest_folder = st.selectbox("Select a folder for this file:", ("Academic", "Journal", "Bills", "Events", "Personal", "Ideas"), key="ingest_file_folder")
    
    if st.button("Ingest File", key="ingest_file_button"):
        if uploaded_file is not None:
            with st.spinner("Ingesting file..."):
                temp_dir = "temp_files"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    result = agent_tools.ingest_document(temp_path, ingest_folder)
                    st.sidebar.success(result)
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.sidebar.error("Please upload a file first.")

    st.sidebar.divider()
    
    text_note = st.text_area("Or, write a new text note (e.g., a journal entry):")
    note_folder = st.selectbox("Select a folder for this note:", ("Journal", "Academic", "Personal", "Ideas"), key="ingest_note_folder")
    
    if st.button("Ingest Note", key="ingest_note_button"):
        if text_note and text_note.strip():
            with st.spinner("Ingesting note..."):
                result = agent_tools.ingest_text_note(text_note.strip(), note_folder)
                st.sidebar.success(result)
        else:
            st.sidebar.error("Please write a note first.")

# --- Global Controls ---
st.sidebar.title("Agent Controls")
st.sidebar.markdown("These actions are independent of your query.")

if st.sidebar.button("New Query (Clear Context)"):
    new_query_session()
    st.toast("New query session started.")

# --- Model Toggle: Ollama vs Local Fine-Tuned ---
st.sidebar.divider()
st.sidebar.markdown("**Model Source**")

# If user pressed detect in previous run, persist suggested values before widgets instantiate
if "pending_detect" in st.session_state and st.session_state.pending_detect:
    st.session_state["insight_local_dir"] = st.session_state.get("detected_insight_dir", st.session_state.get("insight_local_dir", ""))
    st.session_state["pattern_local_dir"] = st.session_state.get("detected_pattern_dir", st.session_state.get("pattern_local_dir", ""))
    st.session_state.pending_detect = False

def _auto_detect_merged_dirs(base_dir: str = "fine_tuning"):
    matches = []
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files and any(f.endswith(".safetensors") for f in files) and ("tokenizer.json" in files or "tokenizer.model" in files):
            matches.append(root)
    insight_guess = next((p for p in matches if "insight" in p.lower()), matches[0] if matches else None)
    pattern_guess = next((p for p in matches if "pattern" in p.lower()), matches[1] if len(matches) > 1 else (matches[0] if matches else None))
    return insight_guess, pattern_guess

if st.sidebar.button("Detect merged dirs", key="detect_merged_dirs"):
    i_guess, p_guess = _auto_detect_merged_dirs()
    if i_guess or p_guess:
        st.session_state["detected_insight_dir"] = i_guess or ""
        st.session_state["detected_pattern_dir"] = p_guess or ""
        # Mark to update widget defaults on next run then rerun
        st.session_state["pending_detect"] = True
        st.rerun()
    else:
        st.warning("No merged model directories found under fine_tuning. Please enter full paths.")

use_local = st.sidebar.toggle("Use local fine-tuned models", value=False, key="use_local_models")
insight_local_dir = st.sidebar.text_input(
    "Insight Writer merged dir",
    value=st.session_state.get("insight_local_dir", "fine_tuning/insight_writer/merged_model_directory"),
    help="Path to merged HF model directory (contains config.json, model.safetensors, tokenizer files)",
    key="insight_local_dir"
)
pattern_local_dir = st.sidebar.text_input(
    "Pattern Miner merged dir",
    value=st.session_state.get("pattern_local_dir", "fine_tuning/pattern_miner_fine-tuned/merged_model_directory"),
    help="Path to merged HF model directory",
    key="pattern_local_dir"
)
if st.sidebar.button("Apply Model Toggle", key="apply_model_toggle"):
    try:
        if use_local:
            agent_tools.set_local_insight_writer_dir(insight_local_dir)
            agent_tools.set_local_pattern_miner_dir(pattern_local_dir)
            st.success("Switched to local fine-tuned models.")
        else:
            agent_tools.set_use_ollama_default()
            st.success("Switched to Ollama llama3.")
    except Exception as e:
        st.error(f"Failed to switch model source: {e}")

# --- View Ingested Documents ---
with st.sidebar.expander("View Ingested Documents", expanded=False):
    if st.button("Refresh Document List", key="refresh_docs"):
        st.rerun()
    
    doc_info = agent_tools.list_ingested_documents()
    
    if "error" in doc_info:
        st.error(doc_info["error"])
    else:
        st.markdown(f"**Total Chunks:** {doc_info['total_chunks']}")
        st.markdown(f"**Total Documents:** {doc_info['total_documents']}")
        
        if doc_info['folders']:
            st.markdown("**By Folder:**")
            for folder, stats in doc_info['folders'].items():
                st.markdown(f"- **{folder}**: {stats['unique_documents']} docs, {stats['chunks']} chunks")
        
        if doc_info['documents']:
            st.markdown("**Document Details:**")
            for i, doc in enumerate(doc_info['documents'][:10]):  # Show first 10
                source_name = os.path.basename(doc['source']) if doc['source'] != 'text_note' else 'Text Note'
                with st.expander(f"{source_name} ({doc['folder']})"):
                    st.write(f"**Source:** {doc['source']}")
                    st.write(f"**Folder:** {doc['folder']}")
                    st.write(f"**Chunks:** {doc['chunks']}")
                    st.write(f"**Ingested:** {doc['ingested_date']}")
                    
                    # Delete button
                    if st.button("Delete Document", key=f"delete_doc_{i}_{doc['source']}", type="secondary"):
                        result = agent_tools.delete_document(doc['source'])
                        if "Error" in result:
                            st.error(result)
                        else:
                            st.success(result)
                            st.rerun()
            
            if len(doc_info['documents']) > 10:
                st.caption(f"... and {len(doc_info['documents']) - 10} more documents")

with st.sidebar.expander("Run Global Pattern Miner"):
    st.markdown("Finds 'big picture' patterns across your data using Map-Reduce analysis.")
    st.markdown("**Select folders to analyze (leave empty to analyze all):**")
    g_folders = st.multiselect(
        "Folders to analyze:", 
        ("Academic", "Journal", "Bills", "Events", "Personal", "Ideas"), 
        default=[], 
        key="global_folders",
        help="Leave empty to analyze all folders. Select specific folders to focus the analysis."
    )
    g_start_date = st.date_input("From:", datetime(2024, 1, 1), key="global_start")
    g_end_date = st.date_input("To:", datetime.now().date(), key="global_end")
    
    if st.button("Run Global Analysis", key="global_analysis_button"):
        with st.chat_message("assistant"):
            with st.spinner("Running Enhanced Map-Reduce Global Miner... This may take a minute."):
                try:
                    # This calls our enhanced Map-Reduce function with folder selection
                    selected_folders = g_folders if g_folders else None
                    global_insight = agent_tools.run_global_stratified_miner(
                        start_date=g_start_date, 
                        end_date=g_end_date,
                        folders=selected_folders
                    )
                    response_content = f"**Global Analysis Result:**\n\n{global_insight}"
                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                except Exception as e:
                    st.error(f"Global Miner failed: {e}")

# --- 5. MAIN CHAT INTERFACE ---

st.title("Personal Cognitive Memory Agent")
st.markdown("Give your query, and the specialized agents will find insights")

# Display chat messages from history (oldest first, like normal chat)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- The "Human Reflection" Buttons ---
if st.session_state.raw_context:
    st.markdown("---")
    st.markdown("**Human Reflection:** What additional insights would you like from this data?")
    
    cols = st.columns(3)
    
    # Check our feature flag to update the button text
    if agent_tools.insight_writer_pipeline is None:
        insight_button_text = "Generate Insight (Fallback Mode)"
    else:
        insight_button_text = "Generate Insight (Fine-Tuned Coach)"

    if cols[0].button(insight_button_text, key="insight_button"):
        with st.chat_message("assistant"):
            with st.spinner(f"Running Insight Writer..."):
                insight = agent_tools.run_insight_writer(st.session_state.raw_context)
                response_content = f"**Empathic Insight:**\n\n{insight}"
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.rerun()

    if cols[1].button("Find Patterns (LLM Analyst)", key="pattern_button"):
        with st.chat_message("assistant"):
            with st.spinner("Running Pattern Miner..."):
                pattern = agent_tools.run_pattern_miner(st.session_state.raw_context)
                response_content = f"**Analytical Pattern:**\n\n{pattern}"
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.rerun()

    if cols[2].button("Visualize", key="viz_button"):
        with st.chat_message("assistant"):
            with st.spinner("Running Visualization Agent..."):
                viz_description, viz_charts = agent_tools.run_viz_agent(st.session_state.raw_context)
                st.markdown(viz_description)
                
                # Store charts in session state for persistence across reruns
                if viz_charts and len(viz_charts) > 0:
                    st.session_state.viz_charts = viz_charts
                    st.session_state.viz_description = viz_description
                    response_content = viz_description + f"\n\n*Displayed {len(viz_charts)} chart(s).*"
                else:
                    st.session_state.viz_charts = []
                    st.session_state.viz_description = viz_description
                    response_content = viz_description
        
        # Add to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.rerun()

# Display stored charts after rerun (before query input section)
if st.session_state.viz_charts and len(st.session_state.viz_charts) > 0:
    st.markdown("---")
    st.markdown("### Visualization Charts")
    st.markdown(st.session_state.viz_description)
    for i, chart in enumerate(st.session_state.viz_charts):
        st.plotly_chart(chart, width='stretch', key=f"viz_chart_{i}")
    st.markdown("---")

# --- The Main Query Input ---
st.markdown("---")
with st.expander("Query Configuration", expanded=True):
    q_folders = st.multiselect("Folders to search:", ("Academic", "Journal", "Bills", "Events", "Personal", "Ideas"), default=["Academic"])
    q_top_k = st.slider("Top-K Chunks:", min_value=1, max_value=20, value=8)
    q_col1, q_col2 = st.columns(2)
    # These are `datetime.date` objects, which our `agent_tools.py` now correctly handles
    q_start_date = q_col1.date_input("Start Date:", datetime(2024, 1, 1), key="query_start")
    q_end_date = q_col2.date_input("End Date:", datetime.now().date(), key="query_end")

if prompt := st.chat_input("Ask a question about your memories..."):
    # Skip empty prompts
    if not prompt or not prompt.strip():
        st.warning("Please enter a non-empty query.")
    else:
        # 1. Clear context from any *previous* query (but keep chat history)
        st.session_state.raw_context = None
        st.session_state.citations = []
        
        # 2. Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. This is the "Agent-as-a-coordinator" logic
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # --- Step A: Query Enhancer (Prompt/response optimiser) ---
                st.write("Running Query Enhancer ('Prompt/response optimiser')...")
                enhancer_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a query optimization specialist for semantic search in personal databases.

**Your Task:**
Transform the user's natural language query into an optimized search query for a vector database that will retrieve the most relevant information.

**Optimization Guidelines:**
1. **Preserve Intent**: Keep the core meaning and intent of the original query
2. **Add Context**: Incorporate folder names and date range naturally into the query
3. **Use Synonyms**: Include alternative terms the user might have meant
4. **Focus Keywords**: Emphasize the most important search terms
5. **Remove Ambiguity**: Clarify vague terms with more specific alternatives
6. **Keep It Natural**: The query should still read naturally, not like a keyword list

**Output:**
Return ONLY the refined search query (1-2 sentences max). Do not add explanations or meta-commentary.
"""),
                    ("user", """Original Query: '{query}'
Relevant Folders: {folders}
Date Range: {start_date} to {end_date}

Refined search query:""")
                ])
                enhancer_chain = enhancer_prompt | llm_enhancer | StrOutputParser()
                enhanced_query = enhancer_chain.invoke({
                    "query": prompt,
                    "folders": q_folders,
                    "start_date": q_start_date.strftime("%Y-%m-%d"),
                    "end_date": q_end_date.strftime("%Y-%m-%d")
                })
                
                # --- Step B: Recall (Retrieval augmented generation) ---
                st.write(f"Running RAG with enhanced query: '{enhanced_query}'")
                
                answer, raw_context, citations = agent_tools.recall_from_folder(
                    query=enhanced_query,
                    folders=q_folders,
                    top_k=q_top_k,
                    start_date=q_start_date, # Pass the date objects directly
                    end_date=q_end_date     # Pass the date objects directly
                )
                
                # --- Step C: Store context & citations for "toggles" ---
                st.session_state.raw_context = raw_context
                st.session_state.citations = citations
                
                # --- Step D: Display the refined answer ---
                st.markdown(answer)
                if citations:
                    with st.expander("View Sources (Citations)"):
                        st.json(st.session_state.citations)
                
                # Add the final answer to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Clear visualization charts when new query is made
        st.session_state.viz_charts = []
        st.session_state.viz_description = ""

        # Re-run the script to show the "Human Reflection" buttons
        st.rerun()
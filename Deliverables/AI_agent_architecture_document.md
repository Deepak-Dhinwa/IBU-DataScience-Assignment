# AI Agent Architecture (Concise)

## 1) What this system does

Think of the Personal Cognitive Memory Agent as your fast, local analyst. It’s a RAG‑based multi‑agent system for personal documents (folders: Academic, Journal, Bills, Events, Personal, Ideas). You ask a question; a coordinator sends it to the right specialists for insights, patterns, cross‑folder analysis, and charts.

Tech we use: Streamlit (UI), ChromaDB (vectors), Ollama (local LLMs), LangChain (orchestration).

---

## 2) Core components (what, how, why)

1. Ingestion (`agent_tools/ingestion.py`)
   - What: Add PDFs/TXT or quick notes, split them smartly, tag with metadata, and store vectors. You can also list and delete items.
   - How: RecursiveCharacterTextSplitter (docs 1000/200, notes 500/100); metadata: folder/source/timestamp; PDF fallback: PyMuPDF → UnstructuredPDFLoader.
   - Why: Reliable parsing across formats; metadata enables folder/date filtering; local‑first privacy.

2. Vector DB (ChromaDB)
   - What: Local, persistent store at `./chroma_db`.
   - How: `nomic-embed-text:latest` embeddings via Ollama; folder and date filters (separate `$gte`/`$lte` in `$and`).
   - Why: Lightweight, fast, private; strong metadata filtering; zero external deps.

3. Retrieval (`agent_tools/retrieval.py`)
   - What: `recall_from_folder(query, folders, top_k, start_date, end_date)` returns (answer, context, citations). No black box—you get what was used.
   - How: Build filters → similarity search → top‑K chunks (UI slider 1–20, default 8) → Refiner synthesizes.
   - Why: Targeted, explainable RAG with citations; robust date filter handling.

4. Query Enhancement (in `app.py`)
   - What: Makes your query retrieval‑friendly (synonyms, clarifies intent, adds folder/date cues).
   - Model: `llama3:latest` via ChatOllama.
   - Why: Improves recall/precision of vector search.

5. Specialized Agents (`agent_tools/agents/`)
   - Insight Writer: Empathetic, concise insights (≤ ~150 words). Uses the fine‑tuned TinyLlama when available, otherwise `llama3`.
   - Pattern Miner: Clear, categorized patterns with evidence (≤ ~150 words). Uses the fine‑tuned TinyLlama when available, otherwise `llama3`.
   - Global Stratified Miner: Cross‑folder Map‑Reduce analysis powered by `llama3`.
   - Visualization Agent: Extracts quantitative JSON and turns it into Plotly charts.

6. Model Management (`agent_tools/models.py`)
   - What: Initializes base LLM, embeddings, and optional local fine‑tuned pipelines (4‑bit if possible, fallback FP32).
   - Why: Single place to switch between Ollama `llama3` and local fine‑tuned TinyLlama via UI toggle and paths.

7. Frontend (`app.py`)
   - What: Upload files or notes; list and delete ingested docs; ask questions; choose folders/dates; pick top‑K; run agents; see charts in a dedicated section.
   - Extras: Sidebar model source toggle; auto‑detect merged dirs; session state for chat, context, charts, citations.

---

## 3) Data flow (end‑to‑end)

Ingestion: file/note → parse (PyMuPDF → Unstructured) → split (1000/200 or 500/100) → tag metadata → embed (`nomic-embed-text`) → store in Chroma.

Query: your question → Query Enhancer (`llama3`) → filtered retrieval (folders/date, top‑K) → Refiner makes a grounded answer + citations → specialized agent builds on that context.

Global analysis: per‑folder Map (5 prompts × up to 3 docs) → folder summaries → Reduce across folders → patterns + summary + recommendations.

---

## 4) Models, datasets, and why

- Base LLM: `llama3:latest` (Ollama)
  - Why: Strong local reasoning, zero API cost, privacy.

- Embeddings: `nomic-embed-text:latest` (Ollama)
  - Why: 8192 context, high quality, local‑first.

- Fine‑tuned (TinyLlama/TinyLlama‑1.1B‑Chat‑v1.0)
  - Insight Writer: 280 examples (generated via Ollama using long, complex contexts)
  - Pattern Miner: 60 examples (generated via Ollama using long, complex contexts)
  - Technique: LoRA (PEFT) with 4‑bit quantization (BitsAndBytes); merged HF checkpoints for local pipelines; optional Ollama conversion.
  - Why: 1.1B runs on modest GPUs; LoRA+4‑bit is efficient; training steers tone and evidence use.

Model switching:

- Sidebar toggle selects: local fine‑tuned merged HF pipelines (primary) or Ollama `llama3` (fallback).
- If 4‑bit load fails, auto‑fallback to full precision.

---

## 5) Agent behaviors (inputs → outputs)

- Insight Writer
  - Input: Refined answer + retrieved context
  - Output: 2–3 short paragraphs, actionable, empathetic; trimmed to limit

- Pattern Miner
  - Input: Retrieved context
  - Output: Patterns categorized as Statistical / Quantitative / Qualitative / Relationships, each with evidence & strength

- Global Stratified Miner
  - Input: Folder scope + date range
  - Output: Cross‑folder patterns, summary bullets, recommendations (≤ ~250 words)

- Visualization Agent
  - Input: Retrieved/processed context
  - Output: 1–3 Plotly charts (pie/bar/line) using flexible JSON keys; robust JSON parsing and per‑chart error isolation

---

## 6) Robustness choices (practical)

- PDF parsing: tiered fallback; handles irregular layouts
- Chroma filters: separate `$gte`/`$lte` with `$and` to avoid syntax pitfalls
- Visualization: tolerant JSON parsing, flexible numeric keys (`value/amount/count/total`), skip bad rows instead of crashing
- Model loading: try 4‑bit first; fallback to FP32; clear errors; continue with `llama3` when needed
- Local‑first: data never leaves machine; no external APIs required

---

## 7) Deployment notes

- Default: Ollama `llama3` + `nomic-embed-text`, ChromaDB persisted at `./chroma_db`.
- Fine‑tuned: load merged HF pipelines from local dirs (set in sidebar). Optional GGUF + Modelfile if converting to Ollama.
- Streamlit UI provides ingestion, querying, agent runs, and deletion; charts rendered outside chat containers and persisted in session state.

---

## 8) What to remember (at a glance)

- RAG + multi‑agent, fully local
- Embeddings: `nomic-embed-text`; Base LLM: `llama3`
- Fine‑tuned TinyLlama via LoRA 4‑bit: Insight (280 ex), Pattern (60 ex)
- Toggle between Ollama and local pipelines; safe fallbacks
- Reliable parsing, filtering, and visualization with robust error handling

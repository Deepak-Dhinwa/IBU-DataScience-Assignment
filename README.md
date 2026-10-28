# Fieldnote

One-page summary: open `poster.html` in a browser.
Start: `.\run.ps1`

**Deepak Dhinwa** · 23B2149 · Mechanical Engineering, IIT Bombay

A local memory for your own documents. Notes go into Chroma, then a small set of agents answers from that store and shows the passage it used. Everything runs on Ollama: `llama3` for writing and `nomic-embed-text` for embeddings. No external LLM API is required.

## What it does

- Ingest a `.pdf` or `.txt`, or a typed note, into one of six folders: Academic, Journal, Bills, Events, Personal, Ideas.
- Retrieve from selected folders and an optional date range, then answer with the retrieved passages.
- Run a pattern miner and an insight writer over the retrieved text, and draw charts when the text has numbers.
- Run a global pass that summarizes folders over a date range.

The Chroma store already on disk is `chroma_db/` (44 chunks across 7 documents). Sample notes are in `sample_data/`. `USE_CUSTOM_FINETUNED_MODEL` in `agent_tools/config.py` is off, so the running app uses `llama3`. The fine-tuned weights under `fine_tuning/` are kept and are not loaded.

## Run

Ollama needs to be serving, with `llama3` and `nomic-embed-text` pulled. From this folder:

```bash
IBU_venv\Scripts\streamlit.exe run app.py
```

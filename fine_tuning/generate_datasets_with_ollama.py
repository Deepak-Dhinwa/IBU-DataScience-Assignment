# python fine_tuning\generate_datasets_with_ollama.py --model llama3:latest --num-insight 280 --num-pattern 60 --log-interval 20

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

from langchain_ollama import ChatOllama

SAMPLE_DIR = Path("./sample_data")
OUT_INSIGHT = Path("./fine_tuning/insight_writer/dataset.json")
OUT_PATTERN = Path("./fine_tuning/pattern_miner/dataset.json")

SYSTEM_INSIGHT = (
    "You are a concise, empathetic insight writer for a personal cognitive memory agent. "
    "Given a raw, potentially large multi-domain personal context, write a single, well-structured, "
    "grounded insight (100-150 words). Be specific, actionable, and avoid fluff. Keep factual grounding."
)

SYSTEM_PATTERN = (
    "You are a Pattern Miner. Given raw, potentially large multi-domain personal context, extract patterns "
    "and categorize them into Statistical, Quantitative, Qualitative, and Relationships. Include brief evidence "
    "snippets from the context. Output a clean, readable markdown section titled '=== PATTERN ANALYSIS ===' with the 4 sections, "
    "a Key Finding, and a Recommendation."
)

USER_TEMPLATE = (
    "Data (raw context):\n" 
    "{context}\n\n"
    "Instructions: Analyze thoroughly. If numeric data exists, incorporate quantitative observations. "
    "If there are multiple domains (academic, finances, projects, events, journal), reflect interconnections."
)

# Longer contexts: we compose multiple files and paragraphs
MIN_CHUNKS = 3
MAX_CHUNKS = 8
MIN_WORDS = 250
MAX_WORDS = 1200


def read_sample_files() -> Dict[str, str]:
    texts = {}
    if not SAMPLE_DIR.exists():
        return texts
    for p in SAMPLE_DIR.glob("**/*.*"):
        if p.suffix.lower() in {".txt", ".md"}:
            try:
                texts[str(p)] = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return texts


def chunk_text(text: str) -> List[str]:
    # simple paragraph-based chunking for variety
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


def build_long_context(all_chunks: List[str]) -> str:
    n = random.randint(MIN_CHUNKS, min(MAX_CHUNKS, max(1, len(all_chunks))))
    picks = random.sample(all_chunks, n) if len(all_chunks) >= n else all_chunks
    ctx = "\n\n".join(picks)
    # enforce min/max words by expanding or trimming
    words = ctx.split()
    if len(words) < MIN_WORDS and all_chunks:
        extra = random.sample(all_chunks, min(len(all_chunks), max(1, (MIN_WORDS - len(words)) // 40)))
        ctx = ctx + "\n\n" + "\n\n".join(extra)
    elif len(words) > MAX_WORDS:
        ctx = " ".join(words[:MAX_WORDS])
    return ctx


def ensure_dirs():
    OUT_INSIGHT.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATTERN.parent.mkdir(parents=True, exist_ok=True)


def generate_examples(llm: ChatOllama, num: int, system_prompt: str, log_interval: int = 25, label: str = "") -> List[Dict]:
    files = read_sample_files()
    if not files:
        raise RuntimeError("No sample_data found. Add .txt/.md files under ./sample_data")

    # pool of chunks
    pool: List[str] = []
    for content in files.values():
        pool.extend(chunk_text(content))
    random.shuffle(pool)

    examples: List[Dict] = []
    for i in range(1, num + 1):
        context = build_long_context(pool)
        prompt = USER_TEMPLATE.format(context=context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        # generate
        try:
            resp = llm.invoke(messages)
            text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            text = f"[GENERATION_ERROR] {e}"
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Data: {context}"},
                {"role": "assistant", "content": text}
            ]
        })
        if log_interval and i % log_interval == 0:
            prefix = f"[{label}] " if label else ""
            print(f"{prefix}Generated {i}/{num} examples...")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate datasets using Ollama with long, complex contexts")
    parser.add_argument("--model", type=str, default="llama3:latest", help="Ollama model to use")
    parser.add_argument("--num-insight", type=int, default=240, help="Number of Insight Writer examples")
    parser.add_argument("--num-pattern", type=int, default=240, help="Number of Pattern Miner examples")
    parser.add_argument("--log-interval", type=int, default=25, help="Print progress every N examples (0 to disable)")
    args = parser.parse_args()

    print(f"Connecting to Ollama model: {args.model}")
    llm = ChatOllama(model=args.model)

    ensure_dirs()

    print("Generating Insight Writer dataset with long/complex contexts...")
    insight = generate_examples(llm, args.num_insight, SYSTEM_INSIGHT, log_interval=args.log_interval, label="INSIGHT")
    OUT_INSIGHT.write_text(json.dumps(insight, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Saved: {OUT_INSIGHT} ({len(insight)} examples)")

    print("Generating Pattern Miner dataset with long/complex contexts...")
    pattern = generate_examples(llm, args.num_pattern, SYSTEM_PATTERN, log_interval=args.log_interval, label="PATTERN")
    OUT_PATTERN.write_text(json.dumps(pattern, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Saved: {OUT_PATTERN} ({len(pattern)} examples)")

    print("Done.")


if __name__ == "__main__":
    main()
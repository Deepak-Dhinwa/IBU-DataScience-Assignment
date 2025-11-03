"""
Model loading and initialization for LLM, embeddings, and fine-tuned models.
"""
import os
import torch
from langchain_ollama import ChatOllama, OllamaEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

from .config import (
    USE_CUSTOM_FINETUNED_MODEL,
    CUSTOM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    ADAPTER_PATH
)

# Global model instances
llm = None
embedding_model = None
insight_writer_pipeline = None
pattern_miner_pipeline = None


def initialize_llm():
    """Initialize the base LLM (llama3:latest)."""
    global llm
    try:
        llm = ChatOllama(model="llama3:latest")
        print("ChatOllama (llama3:latest) loaded successfully.")
    except Exception as e:
        print(f"Error loading ChatOllama: {e}. Is 'ollama serve' running?")
        llm = None
    return llm


def initialize_embedding_model():
    """Initialize the embedding model."""
    global embedding_model
    print(f"Loading Ollama embedding model ({EMBEDDING_MODEL_NAME})...")
    try:
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        print(f"OllamaEmbeddings ({EMBEDDING_MODEL_NAME}) loaded successfully.")
        print(f"Note: nomic-embed-text is excellent - outperforms OpenAI ada-002 and has 8192 context length.")
    except Exception as e:
        print(f"Error loading OllamaEmbeddings: {e}. Is 'ollama serve' running?")
        print("Trying fallback: nomic-embed-text:latest...")
        try:
            embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
            print("Fallback embedding model loaded successfully.")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            embedding_model = None
    return embedding_model


def load_insight_writer_pipeline():
    """
    Loads the fine-tuned Phi-3 model *OR* the custom Ollama model.
    """
    global llm, insight_writer_pipeline
    
    if not USE_CUSTOM_FINETUNED_MODEL:
        print("--- NOTE: USE_CUSTOM_FINETUNED_MODEL=False. Using llama3 fallback. ---")
        return None  # Return None to trigger the fallback logic in run_insight_writer

    # --- Path A: Load the Custom Ollama Model (The FINAL Plan) ---
    # This is for when your 'ollama create' is done.
    try:
        print(f"Attempting to load custom fine-tuned model from Ollama: {CUSTOM_MODEL_NAME}")
        custom_llm = ChatOllama(model=CUSTOM_MODEL_NAME)
        # Test the model
        custom_llm.invoke("test")
        print(f"Successfully loaded custom model '{CUSTOM_MODEL_NAME}'.")
        # Set our *main* llm to be the custom one for all tasks!
        llm = custom_llm
        # We return a *dummy pipeline* just to signal success (since Ollama is now the pipeline)
        insight_writer_pipeline = "OLLAMA_LOADED"
        return "OLLAMA_LOADED"
    except Exception as e:
        print(f"--- WARNING: Could not load '{CUSTOM_MODEL_NAME}' from Ollama. ---")
        print(f"Error: {e}")
        print("Falling back to transformers... (This will likely crash)")

    # --- Path B: Load with Transformers (The OOM-Crash Plan) ---
    # This code is here as our "source code" deliverable.
    if not os.path.exists(ADAPTER_PATH):
        print(f"---!!! WARNING: Adapter path not found: {ADAPTER_PATH}. Using llama3 fallback. !!!---")
        return None
    try:
        print("Loading fine-tuned Insight Writer model via Transformers...")
        base_model_id = "microsoft/Phi-3-mini-4k-instruct"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Loading adapters from {ADAPTER_PATH}...")
        lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        lora_model.eval()
        print("Fine-tuned model (Transformers) loaded successfully.")
        pipeline_instance = pipeline(
            "text-generation",
            model=lora_model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            torch_dtype=torch.bfloat16
        )
        insight_writer_pipeline = pipeline_instance
        return pipeline_instance
    except Exception as e:
        print(f"---!!! FAILED TO LOAD FINE-TUNED MODEL (Likely OOM) !!!---: {e}")
        print("--- Using llama3 fallback. ---")
        insight_writer_pipeline = None
        return None


def initialize_all_models():
    """Initialize all models (LLM, embeddings, and fine-tuned pipeline)."""
    initialize_llm()
    initialize_embedding_model()
    load_insight_writer_pipeline()
    return llm, embedding_model, insight_writer_pipeline


# ---- Local merged model support ----
def _load_local_pipeline_from_dir(model_dir: str, max_new_tokens: int = 256):
    """Load a local merged Hugging Face model directory into a transformers pipeline."""
    if not model_dir or not os.path.isdir(model_dir):
        raise ValueError(f"Local model dir not found: {model_dir}")
    print(f"Loading local merged model from: {model_dir}")
    use_cuda = torch.cuda.is_available()
    model = None
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ) if use_cuda else None
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        )
    except Exception as e:
        print(f"4-bit load failed, falling back to full precision: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
    )
    return pipe


def set_local_insight_writer_dir(model_dir: str):
    """Switch Insight Writer to use a local merged model pipeline."""
    global insight_writer_pipeline
    insight_writer_pipeline = _load_local_pipeline_from_dir(model_dir, max_new_tokens=200)
    print("Insight Writer: using local merged model pipeline.")


def set_local_pattern_miner_dir(model_dir: str):
    """Switch Pattern Miner to use a local merged model pipeline."""
    global pattern_miner_pipeline
    pattern_miner_pipeline = _load_local_pipeline_from_dir(model_dir, max_new_tokens=300)
    print("Pattern Miner: using local merged model pipeline.")


def set_use_ollama_default():
    """Revert to Ollama-backed defaults (llama3 + no local pipelines)."""
    global insight_writer_pipeline, pattern_miner_pipeline
    insight_writer_pipeline = None
    pattern_miner_pipeline = None
    initialize_llm()
    print("Switched back to Ollama defaults (llama3).")


# Initialize models when module is imported
initialize_all_models()


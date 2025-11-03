"""
Insight Writer Agent - Generates empathetic, concise insights from personal data.
Supports both fine-tuned model and fallback to llama3.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .. import models
from ..config import CUSTOM_MODEL_NAME


def run_insight_writer(data_blob: str) -> str:
    """
    Runs the fine-tuned Insight Writer *or* its llama3 fallback. (The "Coach")
    """
    # Always read latest pipeline/llm from models to reflect runtime toggles
    current_pipeline = models.insight_writer_pipeline
    current_llm = models.llm

    if current_pipeline is None:
        # --- FALLBACK PATH (llama3) ---
        print("Insight Writer (Fallback): Running llama3 model.")
        if current_llm is None:
            return "Error: Base LLM is not loaded."
            
        prompt = ChatPromptTemplate.from_messages([
             ("system", """You are a concise, empathetic insight writer. Your job is to extract meaningful personal insights from data - NO FLUFF, NO VERBOSITY.

**RULES:**
- Maximum 2-3 SHORT paragraphs (100-150 words total)
- Be direct and specific - cut out flowery language
- Focus ONLY on what the data actually shows
- If data shows emotional content, mention it briefly
- If data is just factual (course info, events), acknowledge that - don't force emotional insights where none exist
- No generic life coaching advice unless data explicitly warrants it

**Output Format:**
Paragraph 1 (1-2 sentences): What the data reveals about their current state/situation
Paragraph 2 (1-2 sentences): Any patterns, concerns, or positive aspects visible
Paragraph 3 (OPTIONAL, 1 sentence): One brief, actionable takeaway if clear

**BAD EXAMPLE (Too verbose):**
"As I reflect on this data chunk from your memory, I sense a sense of calmness..."

**GOOD EXAMPLE (Concise):**
"Your data shows focus on ME319 course structure and scheduling. The attention to details like homework percentages (10%) and exam breakdowns suggests organized planning. Consider tracking your actual performance against these structures to identify what's working."

**Data:**
{data}"""),
             ("user", "Write a concise (2-3 short paragraphs max), direct insight based on this data. Be specific, avoid fluff.")
        ])
        
        chain = prompt | current_llm | StrOutputParser()
        response = chain.invoke({"data": data_blob})
        
        # Post-process: Trim if too verbose (fallback mode)
        if len(response) > 250:
            print("Insight Writer: Response too verbose, trimming...")
            response = _trim_response(response)
        
        return response

    # --- PRIMARY PATH (Fine-Tuned Model) ---
    # This path runs if the pipeline loaded successfully
    try:
        # This is for the custom Ollama model
        if current_pipeline == "OLLAMA_LOADED":
            print(f"Running custom fine-tuned Ollama model: {CUSTOM_MODEL_NAME}")
            prompt_text = f"Data: {data_blob}\nTask: Write a concise insight (2-3 paragraphs, max 150 words)."
            full_prompt_text = f"""You are a reflective insight agent, trained to be an empathetic coach.
Respond to the following task in your fine-tuned style. Keep it concise (max 150 words).
TASK:
{prompt_text}"""
            chain = ChatPromptTemplate.from_template(full_prompt_text) | current_llm | StrOutputParser()
            response = chain.invoke({})
            
            # Post-process: Trim if too verbose (Ollama fine-tuned mode)
            if len(response) > 250:
                print("Insight Writer (Ollama Fine-tuned): Response too verbose, trimming...")
                response = _trim_response(response)
            
            return response

        # This is for the Transformers/PEFT pipeline
        else:
            print(f"Running fine-tuned Transformers pipeline on: {data_blob[:200]}...")
            prompt = f"<s>[INST] Data: {data_blob}\nTask: Write a concise insight (2-3 paragraphs, max 150 words). [/INST]"
            result = current_pipeline(prompt)
            full_text = result[0]['generated_text']
            response = full_text.split("[/INST]")[-1].strip().replace("</s>", "")
            
            # Post-process: Trim if too verbose (fine-tuned mode)
            if len(response) > 250:
                print("Insight Writer (Fine-tuned): Response too verbose, trimming...")
                if current_llm is not None:
                    response = _trim_response(response)
            
            return response
    except Exception as e:
        print(f"Error running insight writer pipeline: {e}")
        return f"Error: {e}"


def _trim_response(response: str) -> str:
    """Trim verbose responses to max 150 words."""
    trim_prompt = ChatPromptTemplate.from_messages([
        ("system", """Trim this insight to max 150 words. Remove flowery language, keep only specific insights."""),
        ("user", "Trim to max 150 words:\n\n{insight}")
    ])
    # Use latest llm for trimming as well
    trim_chain = trim_prompt | models.llm | StrOutputParser()
    return trim_chain.invoke({"insight": response})


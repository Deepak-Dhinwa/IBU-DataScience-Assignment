"""
Pattern Miner Agent - Extracts and categorizes patterns from data.
Categorizes into Statistical, Quantitative, Qualitative, and Relationships.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .. import models


def run_pattern_miner(data_blob: str) -> str:
    """
    Advanced Pattern Miner with structured multi-step analysis.
    Identifies multiple pattern types: temporal trends, frequency patterns, correlations, and anomalies.
    """
    try:
        print(f"Running Advanced Pattern Miner on: {data_blob[:200]}...")
        # Always read latest runtime models
        current_pipeline = models.pattern_miner_pipeline
        current_llm = models.llm

        # If local pipeline is set, use it directly
        if current_pipeline is not None:
            prompt_text = ChatPromptTemplate.from_messages([
                ("system", """You are a Pattern Miner. Extract and categorize real patterns from the data.\n\n**Categorize Patterns Into:**\n1. Statistical\n2. Quantitative\n3. Qualitative\n4. Relationships\n\nFollow the required output format. Use only evidence from the data."""),
                ("user", "Data:\n{context}\n\nTask: Produce '=== PATTERN ANALYSIS ===' sections with evidence and a Key Finding + Recommendation.")
            ]).format(context=data_blob).to_string()
            result = current_pipeline(prompt_text)
            text = result[0]['generated_text']
            # Heuristic: take tail after last occurrence of the user content marker if present
            response = text
            if len(response) > 300:
                response = _trim_response(response)
            return response

        if current_llm is None:
            return "Error: Base LLM is not loaded."
        
        # Step 1: Structured Analysis Phase - Categorized Patterns
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Pattern Miner. Extract and categorize real patterns from the data.

**Categorize Patterns Into:**
1. **Statistical**: Trends, distributions, frequencies, counts (quantitative measures)
2. **Quantitative**: Numbers, percentages, ratios, measurable metrics explicitly mentioned
3. **Qualitative**: Themes, topics, sentiment, categorical observations
4. **Relationships**: Connections, correlations, dependencies between entities/concepts

**Pattern Format for Each:**
- Name: [Brief pattern name]
- Type: [Statistical/Quantitative/Qualitative/Relationship]
- Evidence: [Specific quote or number from data]
- Strength: [Weak/Moderate/Strong]

**Output Format:**
```
=== PATTERN ANALYSIS ===

**Statistical Patterns:**
- [Pattern Name] (Strength: [Weak/Moderate/Strong])
  Evidence: [specific numbers, trends, counts]

**Quantitative Patterns:**
- [Pattern Name] (Strength: [Weak/Moderate/Strong])
  Evidence: [specific metrics, percentages, ratios]

**Qualitative Patterns:**
- [Pattern Name] (Strength: [Weak/Moderate/Strong])
  Evidence: [specific themes, topics mentioned]

**Relationships:**
- [Pattern Name] (Strength: [Weak/Moderate/Strong])
  Evidence: [how entities relate, correlations]

**Key Finding:**
[1 sentence: Most significant pattern across all categories]

**Recommendation:**
[1 specific action if pattern suggests one, else "None"]
```

**Rules:**
- If no patterns in a category, omit that section
- Evidence MUST be specific quotes/numbers from data
- NO generic patterns - only what's explicitly observable
- Be concise but complete

**Data:**
{context}"""),
            ("user", "Categorize patterns into Statistical, Quantitative, Qualitative, and Relationships. Provide evidence for each.")
        ])
        
        chain = analysis_prompt | current_llm | StrOutputParser()
        response = chain.invoke({"context": data_blob})
        
        # Step 2: Post-processing - Trim verbose outputs
        if len(response) > 300:
            print("Pattern Miner: Response too verbose, trimming...")
            response = _trim_response(response)
        
        return response
    except Exception as e:
        print(f"Error in Pattern Miner: {e}")
        return f"Error in Pattern Miner: {e}"


def _trim_response(response: str) -> str:
    """Trim verbose responses to max 150 words."""
    trim_prompt = ChatPromptTemplate.from_messages([
        ("system", """Trim this pattern analysis to be concise (max 150 words). Remove fluff, generic statements, and repetition. Keep only specific patterns with evidence."""),
        ("user", "Trim this to max 150 words, keeping only real patterns with evidence:\n\n{analysis}")
    ])
    trim_chain = trim_prompt | models.llm | StrOutputParser()
    return trim_chain.invoke({"analysis": response})


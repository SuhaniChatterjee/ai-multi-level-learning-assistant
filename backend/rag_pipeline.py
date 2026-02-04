from backend.prompt_templates import get_prompt_template
from backend.llm_engine import generate_response


def build_prompt(question: str, context: str, level: str) -> str:
    """
    Builds the final prompt sent to the LLM.
    """

    instructions = get_prompt_template(level)

    prompt = f"""
You are a helpful AI tutor.

Instructions:
{instructions}

Document Context:
{context}

User Question:
{question}

Answer:
"""
    return prompt


def run_rag_pipeline(question: str, context: str, difficulty_level: str) -> str:
    """
    Main RAG pipeline:
    - validates input
    - builds prompt
    - sends prompt to LLM
    """

    if not question.strip():
        return "Please enter a valid question."

    if not context.strip():
        return "No relevant information found in the uploaded documents."

    prompt = build_prompt(question, context, difficulty_level)
    return generate_response(prompt)

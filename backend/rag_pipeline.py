from backend.prompt_templates import get_prompt_template
from backend.llm_engine import generate_response


def run_rag_pipeline(
    question: str,
    context: str,
    difficulty_level: str
) -> str:
    """
    Combines retrieved context, difficulty-level prompt,
    and user question to generate a final answer.
    """

    instruction = get_prompt_template(difficulty_level)

    final_prompt = f"""
You are a helpful AI tutor.

Instructions:
{instruction}

Context (use ONLY this information):
{context}

Question:
{question}

Answer:
"""

    response = generate_response(final_prompt)
    return response

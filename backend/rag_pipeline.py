from backend.prompt_templates import get_prompt_template
from backend.llm_engine import generate_response, rephrase_question
from backend.vector_store import search_vector_store

def run_rag_pipeline(
    question: str,
    difficulty_level: str,
    chat_history: list = None
) -> tuple[str, str]:
    """
    Retrieves context from the FAISS vector store,
    combines it with the difficulty-level prompt,
    and user question to generate a final answer.
    
    Returns:
        tuple containing (LLM response, retrieved context)
    """
    
    # 1. Retrieve context chunks from Vector Store
    try:
        search_query = question
        if chat_history:
            search_query = rephrase_question(chat_history, question)
            import logging
            logging.info(f"Rephrased question for retrieval: {search_query}")
            
        retrieved_chunks = search_vector_store(search_query, top_k=3)
        context = "\n\n---\n\n".join(retrieved_chunks)
    except FileNotFoundError:
        return "[Error] Knowledge base not found. Please upload a PDF first.", ""

    if not context:
        return "I couldn't find any relevant information in the uploaded documents to answer your question.", ""

    # 2. Build prompt
    instruction = get_prompt_template(difficulty_level)

    final_prompt = f"""
You are a helpful AI tutor.

Instructions:
{instruction}

Context (use ONLY this information to answer):
{context}

Question:
{question}

Answer:
"""

    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Generating {difficulty_level} explanation for query.")
    
    # 3. Generate Answer
    response = generate_response(final_prompt, chat_history)
    logging.info("Explanation generated successfully.")
    return response, context

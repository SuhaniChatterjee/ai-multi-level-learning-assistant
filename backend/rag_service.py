from backend.rag_pipeline import run_rag_pipeline
from data.vector_store import get_relevant_context


def answer_question(question: str, level: str):

    # retrieve context from vector DB
    context = get_relevant_context(question)

    # run rag pipeline
    answer = run_rag_pipeline(
        question=question,
        context=context,
        difficulty_level=level
    )

    return answer
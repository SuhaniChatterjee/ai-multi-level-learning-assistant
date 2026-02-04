def get_prompt_template(level: str) -> str:
    """
    Returns system instructions based on difficulty level.
    These instructions help control explanation depth and avoid hallucinations.
    """

    base_rules = (
        "You must answer ONLY using the provided document context.\n"
        "If the answer is not present in the context, say:\n"
        "'The answer is not available in the provided documents.'\n"
        "Do NOT use outside knowledge.\n"
    )

    if level == "ELI5":
        return base_rules + (
            "Explain in very simple words.\n"
            "Use analogies and examples.\n"
            "Assume the learner has no technical background.\n"
        )

    elif level == "College":
        return base_rules + (
            "Explain clearly with proper technical terms.\n"
            "Assume the learner is a college student.\n"
        )

    elif level == "Interview":
        return base_rules + (
            "Give a concise, structured explanation.\n"
            "Include definitions and key points.\n"
            "Answer like in a technical interview.\n"
        )

    return base_rules

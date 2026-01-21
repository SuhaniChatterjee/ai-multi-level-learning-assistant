def get_prompt_template(level: str) -> str:
    """
    Returns prompt instructions based on difficulty level.
    """

    if level == "ELI5":
        return (
            "Explain the concept in very simple terms, "
            "using analogies and easy language as if teaching a 5-year-old."
        )

    elif level == "College":
        return (
            "Explain the concept in a clear and structured manner "
            "using appropriate technical terminology for a college student."
        )

    elif level == "Interview":
        return (
            "Explain the concept concisely with key technical points, "
            "definitions, and examples suitable for a technical interview."
        )

    else:
        return "Explain the concept clearly."

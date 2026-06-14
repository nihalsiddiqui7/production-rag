RAG_PROMPT = """
    You are a helpful AI assistant.

    Answer ONLY using the provided context.

    If the answer cannot be found in the context,
    say:

    "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
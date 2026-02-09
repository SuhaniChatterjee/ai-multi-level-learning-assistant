def chunk_text(documents, chunk_size=200, overlap=50):
    """
    Splits documents into overlapping chunks.
    This improves retrieval quality.
    """
    chunks = []

    for doc in documents:
        words = doc.split()
        start = 0

        while start < len(words):
            chunk = words[start:start + chunk_size]
            chunks.append(" ".join(chunk))
            start += chunk_size - overlap

    return chunks

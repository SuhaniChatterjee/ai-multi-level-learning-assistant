from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example documents (replace later with real docs)
documents = [
    "Docker is a containerization platform.",
    "Containers package applications with their dependencies.",
    "Containers are lightweight and portable."
]

# Create embeddings
embeddings = model.encode(documents)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


def get_relevant_context(question: str, k: int = 2) -> str:
    """
    Retrieve the most relevant document chunks for a question.
    """

    query_embedding = model.encode([question])

    distances, indices = index.search(np.array(query_embedding), k)

    results = [documents[i] for i in indices[0]]

    return "\n".join(results)
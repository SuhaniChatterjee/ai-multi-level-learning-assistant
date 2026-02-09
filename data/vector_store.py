import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from data.document_loader import load_documents
from data.text_chunker import chunk_text


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []

    def build_index(self, chunks):
        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        self.text_chunks = chunks

    def query(self, question, top_k=3):
        question_embedding = self.model.encode([question])
        question_embedding = np.array(question_embedding).astype("float32")

        distances, indices = self.index.search(question_embedding, top_k)

        return [self.text_chunks[i] for i in indices[0]]


def get_relevant_context(question, top_k=3):
    """
    Main function used by RAG backend.
    Returns relevant document context as a single string.
    """
    documents = load_documents()
    chunks = chunk_text(documents)

    vector_store = VectorStore()
    vector_store.build_index(chunks)

    results = vector_store.query(question, top_k)
    return "\n".join(results)

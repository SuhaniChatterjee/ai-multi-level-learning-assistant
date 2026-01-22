import faiss
from sentence_transformers import SentenceTransformer
import numpy as np


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []

    def build_index(self, chunks: list[str]):
        """
        Creates FAISS index from text chunks.
        """
        self.text_chunks = chunks
        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def query(self, question: str, top_k: int = 3) -> list[str]:
        """
        Retrieves top-k relevant text chunks for a given query.
        """
        question_embedding = self.model.encode([question])
        question_embedding = np.array(question_embedding).astype("float32")

        distances, indices = self.index.search(question_embedding, top_k)

        return [self.text_chunks[i] for i in indices[0]]
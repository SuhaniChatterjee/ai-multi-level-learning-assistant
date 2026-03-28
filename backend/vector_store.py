import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load a lightweight embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index.bin")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.pkl")

def create_vector_store(chunks: list[str]) -> faiss.IndexFlatL2:
    """
    Embeds text chunks and stores them in a FAISS index.
    Saves the index and the corresponding chunks to disk.
    """
    if not chunks:
        raise ValueError("No chunks provided to create the vector store.")

    # 1. Generate embeddings
    embeddings = model.encode(chunks)
    embedding_dim = embeddings.shape[1]

    # 2. Initialize FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    
    # 3. Add embeddings to the index
    index.add(np.array(embeddings, dtype=np.float32))
    
    # 4. Save to disk
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    faiss.write_index(index, VECTOR_STORE_PATH)
    
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index

def load_vector_store() -> tuple[faiss.IndexFlatL2, list[str]]:
    """
    Loads the FAISS index and the corresponding chunks from disk.
    """
    if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("Vector store or chunks file not found. Please create them first.")
    
    index = faiss.read_index(VECTOR_STORE_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
        
    return index, chunks

def search_vector_store(query: str, top_k: int = 3) -> list[str]:
    """
    Embeds a query, searches the FAISS index, and returns the top_k most relevant chunks.
    """
    index, chunks = load_vector_store()
    
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    
    results = []
    for i in indices[0]:
        if i != -1 and i < len(chunks):
            results.append(chunks[i])
            
    return results

if __name__ == "__main__":
    pass

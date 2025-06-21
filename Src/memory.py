
# Src/memory.py
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths
MEMORY_DIR = "./faiss_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)
INDEX_PATH = os.path.join(MEMORY_DIR, "chat.index")
DATA_PATH = os.path.join(MEMORY_DIR, "chat.pkl")

# Load or initialize FAISS index
if os.path.exists(INDEX_PATH) and os.path.exists(DATA_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(DATA_PATH, "rb") as f:
        memory_data = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)  # 384 = all-MiniLM-L6-v2 dim
    memory_data = []

def save_memory():
    faiss.write_index(index, INDEX_PATH)
    with open(DATA_PATH, "wb") as f:
        pickle.dump(memory_data, f)

def add_to_memory(query, response):
    """Add user query and assistant response as a memory pair."""
    text = f"User: {query}\nAssistant: {response}"
    embedding = embedding_model.encode([text])[0].astype("float32")
    index.add(np.array([embedding]))
    memory_data.append(text)
    save_memory()

def retrieve_memory(query, top_k=3):
    """Retrieve similar past memory chunks based on the query."""
    if len(memory_data) == 0:
        return []

    query_embedding = embedding_model.encode([query])[0].astype("float32")
    D, I = index.search(np.array([query_embedding]), top_k)
    return [memory_data[i] for i in I[0] if i < len(memory_data)]

def clear_memory():
    """Clear all memory documents."""
    global index, memory_data
    index = faiss.IndexFlatL2(384)
    memory_data = []
    save_memory()

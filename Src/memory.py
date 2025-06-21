
# Src/memory.py
import os
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Define memory directory
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "chat_memory"

# Initialize new persistent client
client = PersistentClient(path=CHROMA_DIR)

# Define embedding function (replace with your preferred model if needed)
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get collection
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)

# --- Memory Functions ---

def add_to_memory(query, response):
    """Add user query and assistant response as a memory pair."""
    doc_id = f"mem_{len(collection.get()['ids']) + 1}"
    collection.add(
        documents=[f"User: {query}\nAssistant: {response}"],
        ids=[doc_id],
        metadatas=[{"source": "chat"}]
    )

def retrieve_memory(query, top_k=3):
    """Retrieve similar past memory chunks based on the query."""
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0] if results["documents"] else []

# --- Clear all stored memory (for debugging or reset) ---
def clear_memory():
    """Clear all memory documents in the collection."""
    collection.delete(where={"source": {"$eq": "chat"}})


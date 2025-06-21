from pathlib import Path

# Define the output path for the graph_builder.py module
src_dir = Path(r"C:\Users\Asus\Downloads\GraphRAG Project\Src")
src_dir.mkdir(exist_ok=True)
graph_builder_path = src_dir / "graph_builder.py"

# Now generate the updated graph_builder.py module using your supplied logic

graph_builder_custom_code = r"""
import os
import re
import fitz  # PyMuPDF
import contractions
from tqdm import tqdm
import nltk

nltk.download('punkt')
nltk.download('punkt-tab')

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    full_text = ""
    for page in tqdm(doc):
        full_text += page.get_text()
    doc.close()
    return full_text


def clean_text_debug(text):
    removed = {}
    text = text.replace("U.S.", "___US___")
    text = text.replace("U.K.", "___UK___")
    text = contractions.fix(text)
    text = text.replace("___US___", "U.S.")
    text = text.replace("___UK___", "U.K.")

    urls = re.findall(r'http\S+|www\S+|https\S+', text)
    removed['urls'] = urls
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    emails = re.findall(r'\S+@\S+', text)
    removed['emails'] = emails
    text = re.sub(r'\S+@\S+', '', text)

    text = re.sub(r'\n{2,}', '\n---\n', text)
    text = re.sub(r'(?<=\w)\n(?=\w)', ' ', text)

    pattern_to_keep = r"[^\w\s.,;:()?!\-/\'���\"$�]"
    non_alpha = re.findall(pattern_to_keep, text)
    removed['non_alpha'] = non_alpha
    text = re.sub(pattern_to_keep, '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text, removed


def save_cleaned_text(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Cleaned text saved to {output_path}")


def split_into_sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)
"""

# Save to graph_builder.py
with open(graph_builder_path, "w", encoding="utf-8") as f:
    f.write(graph_builder_custom_code)

print(f"Graph Builder Module created at: {graph_builder_path}")

# Define the retriever.py module path
retriever_path = src_dir / "retriever.py"

# Construct retriever module using the user's Neo4j-based querying logic
retriever_code = """
import os
import streamlit as st
from neo4j import GraphDatabase

# Load secrets from .streamlit/secrets.toml
HF_TOKEN = st.secrets["HF_TOKEN"]
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
NEO4J_DATABASE = st.secrets.get("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def retrieve_relevant_triplets(entities):
    query = \"""
    MATCH (a)-[r]->(b)
    WHERE ANY(e IN $entities WHERE toLower(a.name) CONTAINS toLower(e) OR toLower(b.name) CONTAINS toLower(e))
    RETURN a.name AS subject, type(r) AS relation, b.name AS object
    LIMIT 30
    \"""
    with driver.session() as session:
        result = session.run(query, entities=entities)
        return [f"{row['subject']} {row['relation']} {row['object']}" for row in result]


def extract_entities_from_question(question):
    # Very simple token-based entity extraction; can be replaced with spaCy NER
    return question.lower().split()
"""

# Write to retriever.py
with open(retriever_path, "w", encoding="utf-8") as f:
    f.write(retriever_code)

print(f"Retriever module created at: {retriever_path}")

# Define the path for prompt_injector.py
prompt_injector_path = src_dir / "prompt_injector.py"

prompt_injector_code = """
import json
import requests
import streamlit as st

HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://router.huggingface.co/sambanova/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

def build_prompt(context_triplets, question):
    context = (
        "\\n".join(f"- {triplet}" for triplet in context_triplets)
        if context_triplets
        else "No relevant facts were found in the knowledge graph."
    )
    return f\"\"\"You are a helpful assistant that answers questions using a knowledge graph. 
Use only the facts provided below. If the facts do not fully support an answer, clearly state 
that you don't have enough information, but try to communicate any partial insight in a natural, human tone.

Your goal is to be clear, concise, and friendly—like a knowledgeable tutor explaining the answer. 
Do **not** assume, speculate, or make up any information not present in the facts.

Facts:
{context}

Question: {question}
Answer:\"\"\"

def format_messages(prompt):
    return [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable and concise assistant trained to answer questions using only "
                "a given set of knowledge graph facts. You explain clearly and naturally, as if you're "
                "teaching a human, but you must not make up or assume anything beyond the provided facts."
                "If the facts are insufficient, say so directly — but feel free to point out any partial "
                "information that might help the user, while staying strictly factual."
                "Avoid repetition, guesswork, or speculation. Maintain a friendly and helpful tone at all times."
            ),
        },
        {"role": "user", "content": prompt},
    ]

def query_llm_stream(messages):
    payload = {
        "model": "Meta-Llama-3.2-3B-Instruct",
        "messages": messages,
        "stream": True,
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload, stream=True)
    for line in response.iter_lines():
        if not line.startswith(b"data:"):
            continue
        if line.strip() == b"data: [DONE]":
            return
        yield json.loads(line.decode("utf-8").lstrip("data:").rstrip("/n"))
"""

# Save the code to prompt_injector.py
with open(prompt_injector_path, "w", encoding="utf-8") as f:
    f.write(prompt_injector_code)

print(f"Prompt injector module created at: {prompt_injector_path}")

# Define the app.py path
memory_path = src_dir / "memory.py"

# Streamlit chatbot app using the created modules
memory_code = """
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
    \"""Add user query and assistant response as a memory pair.\"""
    text = f"User: {query}\\nAssistant: {response}"
    embedding = embedding_model.encode([text])[0].astype("float32")
    index.add(np.array([embedding]))
    memory_data.append(text)
    save_memory()

def retrieve_memory(query, top_k=3):
    \"""Retrieve similar past memory chunks based on the query.\"""
    if len(memory_data) == 0:
        return []

    query_embedding = embedding_model.encode([query])[0].astype("float32")
    D, I = index.search(np.array([query_embedding]), top_k)
    return [memory_data[i] for i in I[0] if i < len(memory_data)]

def clear_memory():
    \"""Clear all memory documents.\"""
    global index, memory_data
    index = faiss.IndexFlatL2(384)
    memory_data = []
    save_memory()
"""

# Save app.py
with open(memory_path, "w", encoding="utf-8") as f:
    f.write(memory_code)

print(f"Memory Module created at: {memory_path}")

# Define the app.py path
app_path = Path("C:/Users/Asus/Downloads/GraphRAG Project/app.py")

# Streamlit chatbot app using the created modules
app_code = """
import json
import streamlit as st
from datetime import datetime
from Src.retriever import extract_entities_from_question, retrieve_relevant_triplets
from Src.prompt_injector import build_prompt, format_messages, query_llm_stream
from Src.memory import add_to_memory, retrieve_memory, clear_memory

# --- Access secrets ---
HF_TOKEN = st.secrets["HF_TOKEN"]
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in secrets.toml")
else:
    print("HF_TOKEN loaded successfully")

NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
NEO4J_DATABASE = st.secrets.get("NEO4J_DATABASE", "neo4j")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError("One or more Neo4j credentials are missing in secrets.toml")
else:
    print("Neo4j credentials loaded successfully")

# --- Page Config ---
st.set_page_config(page_title="🧠 KG Chatbot", layout="wide")

st.markdown(\"""
    <style>
        [data-testid="stSidebar"] {
            width: 320px;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 320px;
        }
    </style>
\""", unsafe_allow_html=True)

# --- Session Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "all_triples" not in st.session_state:
    st.session_state.all_triples = []

# --- Sidebar ---
with st.sidebar:
    st.title("🛠️ Chatbot Controls")
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.session_state.all_triples = []
        st.rerun()

    st.markdown("### ℹ️ System Info")
    st.markdown("**Model:** `Meta-LLaMA-3B-Instruct`")
    st.markdown("**KG Backend:** `Neo4j`")
    st.markdown(f"**Session Triples:** `{len(st.session_state.all_triples)}`")
    st.markdown("---")

    st.markdown("### 📂 Save & Load Chat")

    if st.session_state.messages:
        chat_data = {
            "messages": st.session_state.messages,
            "triples": st.session_state.all_triples
        }
        st.download_button(
            label="📅 Download Chat as JSON",
            data=json.dumps(chat_data, indent=2),
            file_name="chat_history.json",
            mime="application/json"
        )

    uploaded_file = st.file_uploader("📄 Load Previous Chat (.json)", type="json")
    if uploaded_file is not None:
        try:
            content = json.load(uploaded_file)
            st.session_state.messages = content.get("messages", [])
            st.session_state.all_triples = content.get("triples", [])
            st.success("✅ Chat loaded successfully!")
            st.success("Remove the loaded file from the sidebar to proceed further.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to load chat: {e}")

    st.caption("Developed by - Mohit Gupta")

# --- Header ---
st.markdown("## 📘 Knowledge Graph-Powered Chatbot")
st.markdown("Ask legal or policy-related questions based on the '**eBay User Agreement Knowledge Graph**'.")
st.markdown("---")

# --- Display Chat History ---
for msg in st.session_state.get("messages", []):
    with st.chat_message(msg.get("role", "user")):
        timestamp = msg.get("time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"**🕒 {timestamp}**")
        st.markdown(msg.get("content", ""))

        if msg.get("role") == "assistant":
            if msg.get("memory"):
                with st.expander("📘 Retrieved Memory", expanded=False):
                    for m in msg["memory"]:
                        st.markdown(f"- `{m}`")

            if msg.get("triples"):
                with st.expander("🧠 Retrieved Triples (KG)", expanded=False):
                    for t in msg["triples"]:
                        st.markdown(f"- `{t}`")

# --- Handle User Input ---
def handle_user_input(prompt):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "time": timestamp
    })

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving KG and memory + generating answer..."):
            entities = extract_entities_from_question(prompt)
            triples = retrieve_relevant_triplets(entities)

            for t in triples:
                if t not in st.session_state.all_triples:
                    st.session_state.all_triples.append(t)

            memory_context = retrieve_memory(prompt)
            context = memory_context + triples

            if context:
                with st.expander("📘 Retrieved Context", expanded=False):
                    for t in context:
                        st.markdown(f"- `{t}`")
            else:
                st.warning("No relevant memory or triples found.")

            full_prompt = build_prompt(context, prompt)
            messages = format_messages(full_prompt)

            response_text = ""
            response_placeholder = st.empty()
            for chunk in query_llm_stream(messages):
                content = chunk["choices"][0]["delta"].get("content", "")
                response_text += content
                response_placeholder.markdown(response_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "triples": triples,
            "memory": memory_context
        })

        add_to_memory(prompt, response_text)

# --- Input Field ---
if prompt := st.chat_input("💬 Ask your question here..."):
    handle_user_input(prompt)
"""

# Save app.py
with open(app_path, "w", encoding="utf-8") as f:
    f.write(app_code)

print(f"Streamlit app created at: {app_path}")
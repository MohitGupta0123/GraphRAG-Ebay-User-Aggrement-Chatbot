
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

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 320px;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 320px;
        }
    </style>
""", unsafe_allow_html=True)

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

    # if st.session_state.all_triples:
    #     with st.expander("📌 All Retrieved Triples This Session", expanded=False):
    #         for t in st.session_state.all_triples:
    #             st.markdown(f"- `{t}`")
    
    # st.markdown("---")
    
    st.markdown("### 💾 Save & Load Chat")

    # --- Download Chat ---
    if st.session_state.messages:
        chat_data = {
            "messages": st.session_state.messages,
            "triples": st.session_state.all_triples
        }
        st.download_button(
            label="📥 Download Chat as JSON",
            data=json.dumps(chat_data, indent=2),
            file_name="chat_history.json",
            mime="application/json"
        )

    # --- Upload Chat ---
    uploaded_file = st.file_uploader("📤 Load Previous Chat (.json)", type="json")
    if uploaded_file is not None:
        try:
            content = json.load(uploaded_file)
            st.session_state.messages = content.get("messages", [])
            st.session_state.all_triples = content.get("triples", [])
            st.success("✅ Chat loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to load chat: {e}")

    st.caption("Developed for Amlgo Labs AI Engineer Assignment")


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

        if msg.get("role") == "assistant" and msg.get("triples"):
            with st.expander("📄 Retrieved Triples", expanded=False):
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
                with st.expander("📄 Retrieved Triples", expanded=False):
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
            "triples": triples
        })

        add_to_memory(prompt, response_text)

# --- Input Field ---
if prompt := st.chat_input("💬 Ask your question here..."):
    handle_user_input(prompt)

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
        "\n".join(f"- {triplet}" for triplet in context_triplets)
        if context_triplets
        else "No relevant facts were found in the knowledge graph."
    )
    return f"""You are a helpful assistant that answers questions using a knowledge graph. 
Use only the facts provided below. If the facts do not fully support an answer, clearly state 
that you don't have enough information, but try to communicate any partial insight in a natural, human tone.

Your goal is to be clear, concise, and friendly—like a knowledgeable tutor explaining the answer. 
Do **not** assume, speculate, or make up any information not present in the facts.

Facts:
{context}

Question: {question}
Answer:"""

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

def query_llm_stream(messages, token):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": "Meta-Llama-3.2-3B-Instruct",
        "messages": messages,
        "stream": True,
    }
    response = requests.post(API_URL, headers=headers, json=payload, stream=True)
    for line in response.iter_lines():
        if not line.startswith(b"data:"):
            continue
        if line.strip() == b"data: [DONE]":
            return
        yield json.loads(line.decode("utf-8").lstrip("data:").rstrip("/n"))


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
    query = """
    MATCH (a)-[r]->(b)
    WHERE ANY(e IN $entities WHERE toLower(a.name) CONTAINS toLower(e) OR toLower(b.name) CONTAINS toLower(e))
    RETURN a.name AS subject, type(r) AS relation, b.name AS object
    LIMIT 30
    """
    with driver.session() as session:
        result = session.run(query, entities=entities)
        return [f"{row['subject']} {row['relation']} {row['object']}" for row in result]


def extract_entities_from_question(question):
    # Very simple token-based entity extraction; can be replaced with spaCy NER
    return question.lower().split()

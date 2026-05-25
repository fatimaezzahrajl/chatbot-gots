import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("BAAI/bge-m3")
groq_client = Groq(api_key=GROQ_API_KEY)

st.title("💬 RAG - GOTS")
question = st.text_input("Posez votre question :")

if question:
    vecteur = model.encode(question).tolist()
    resultats = client.search(
        collection_name="gots",
        query_vector=vecteur,
        limit=5
    )
    contexte = "\n".join([r.payload["texte"] for r in resultats])
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Tu es un assistant. Réponds en français sur ce rapport GOTS."},
            {"role": "user", "content": f"Contexte:\n{contexte}\n\nQuestion: {question}"}
        ]
    )
    st.write(response.choices[0].message.content)

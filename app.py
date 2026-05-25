import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
from qdrant_client.models import Filter

# Configuration des secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialisation des clients
@st.cache_resource
def init_clients():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    model = SentenceTransformer("BAAI/bge-m3")
    groq_client = Groq(api_key=GROQ_API_KEY)
    return client, model, groq_client

client, model, groq_client = init_clients()

st.title("💬 RAG - GOTS")
question = st.text_input("Posez votre question :")

if question:
    with st.spinner("Recherche en cours..."):
        # 1. Embedding de la question
        vecteur = model.encode(question).tolist()
        
        # 2. Recherche dans Qdrant
        resultats = client.query_points(
            collection_name="gots",
            query=vecteur,
            limit=5
        )
        
        # 3. Extraire les textes des résultats
        contexte = "\n".join([point.payload["texte"] for point in resultats.points])
        
        # 4. Génération de la réponse
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Tu es un assistant expert en certification GOTS. Réponds toujours en français de manière précise et concise."},
                {"role": "user", "content": f"Contexte:\n{contexte}\n\nQuestion: {question}"}
            ]
        )
        
        st.write(response.choices[0].message.content)

#in the build index side, we prepare the chunk's embeddings using mini lm and the index of these chunks using indexlp for future comparison. This index lp defines the inner product function for cosine similarity later for the chunk embeddings and keeps it ready. In retriever embedding, we get the food decision from decision engine---> rag.explainer---->retriever.py and here it encodes the food names using mini lm and compares it with the faiss index after normalizing it.

import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


# ---------- LOAD ----------
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(str(MODEL_DIR / "food_knowledge.index"))

with open(MODEL_DIR / "food_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(MODEL_DIR / "food_chunk_meta.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ---------- RETRIEVAL ----------
def retrieve_food_context(food_name, reason_tags, k=20): #retrieves top 20 chunks similar to the food query below

    query = f"{food_name} Indian dish origin taste ingredients cuisine" #this is for the info/keywords about the retrieved foods to the Minilm understanding

    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb) #l2 norm of the query for calculating cosine similarity

    scores, idx = index.search(q_emb, k) #cosine similarity

    results = []

    # appen chunks belonging to the same food
    for i in idx[0]:
        if food_name.lower() in metadata[i]["food_name"].lower(): #filters the chunks that has the food name in it
            results.append(chunks[i])

    # fallback if none found
    if len(results) == 0:
        results = [chunks[i] for i in idx[0]]

    return results[:10] #only 2 will be returned for each food but this is for future entension
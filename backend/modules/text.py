#This file is not used anywhere in this system. This is the first baseline text embedding generation file that was build to test the system in the notebook env.
#This python file does not contain any rules or decision making components rather it can be considered as a legacy code that was first written.

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ================= LOAD DATA =================
df = pd.read_csv(os.path.join(DATA_DIR, "food_metadata.csv"))
embeddings = np.load(os.path.join(MODEL_DIR, "minilm_embeddings.npy")) #This embeddings where generated in a notebook environment and the embeddings numpy file
#were stored in this same environment.

model = SentenceTransformer("all-MiniLM-L6-v2") #a sentence transformer model

# ================= FUNCTION =================
def semantic_search(query, top_k=10): #cosine similarity and retrieves top 10
    query = query.lower().strip()
    query_vec = model.encode([query])

    scores = cosine_similarity(query_vec, embeddings)[0]
    top_idx = scores.argsort()[::-1][:top_k]

    results = df.iloc[top_idx].copy()
    results["score"] = scores[top_idx]

    return results
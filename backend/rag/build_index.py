#this is the database building code for faiss-rag. We build 2 chunks for each food maiing it as 700 chunks totally. The chunks are build and stored in three files- food_meta_raw jason, food_sentences and faiss embeddings- food_index. The food_index file is built by faiss IndexLP function to prepare it ready for the cosine similarity later. We use this file to retrieve knowledge about the decision engine retrieved food. MiniLM converts the food embeddings and faiss prepares it for the search process.

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import json

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "food_metadata.csv"
INDEX_DIR = BASE_DIR / "models"

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv(DATA_PATH)

# ---------------- BUILD CHUNKS ----------------
def build_chunks(row):

    name = row.food_name
    chunks = []

    # structured description
    chunks.append(
        f"{name} is a {row.veg_nonveg} {row.course_type} from {row.state} in {row.region} with a {row.taste} taste."
    )

    # main description
    if isinstance(row.description, str):
        sentences = row.description.split(".")
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                chunks.append(f"{name}: {s}")

    return chunks

all_chunks = []
metadata = []

for _, row in df.iterrows():
    chunks = build_chunks(row)
    for c in chunks:
        all_chunks.append(c)
        metadata.append({
            "food_name": row.food_name,
            "state": row.state,
            "region": row.region,
            "course_type": row.course_type,
            "taste": row.taste,
            "veg_nonveg": row.veg_nonveg,
            "description": row.description
        })

# ---------------- EMBEDDING ----------------
embeddings = model.encode(all_chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# normalize (for cosine search)
faiss.normalize_L2(embeddings) 

# ---------------- INDEX ----------------
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # cosine similarity for retriever.py file
index.add(embeddings)

# ---------------- SAVE ----------------
faiss.write_index(index, str(INDEX_DIR / "food_knowledge.index"))

with open(INDEX_DIR / "food_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

with open(INDEX_DIR / "food_chunk_meta.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Index built!")
print("Total chunks:", len(all_chunks))
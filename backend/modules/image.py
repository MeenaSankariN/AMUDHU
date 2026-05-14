import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ASSET_DIR = BASE_DIR / "assets" / "food_images"

# ================= LOAD DATA =================
df = pd.read_csv(DATA_DIR / "food_metadata.csv")
image_embeddings = np.load(MODEL_DIR / "dino_embeddings.npy")

index_to_food = {i: row["image_filename"] for i, row in df.iterrows()}
# mapping filename -> index
embedding_lookup = {row["image_filename"]: i for i, row in df.iterrows()}

# ================= LOAD SAME MODEL USED FOR EMBEDDING =================
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
model.eval()

# ================= EMBEDDING =================
def extract_query_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:,0]

    return emb.squeeze().cpu().numpy()

# ================= DATASET IMAGE SEARCH (HYBRID) =================
from modules.text import embeddings as text_embeddings

def find_similar_images(image_index, top_k=5, region=None, diet=None, course=None):

    query_vec = image_embeddings[image_index].reshape(1,-1)

    candidate_df = df.copy() #copying metadata to work on it to find similarity 

    if region:
        candidate_df = candidate_df[candidate_df["region"] == region]
    if diet:
        candidate_df = candidate_df[candidate_df["veg_nonveg"] == diet]
    if course:
        candidate_df = candidate_df[candidate_df["course_type"] == course]

    if len(candidate_df) == 0:
        return df.iloc[[image_index]]

    subset_embeddings = image_embeddings[candidate_df.index]
    visual_scores = cosine_similarity(query_vec, subset_embeddings)[0]

    candidate_df = candidate_df.copy()
    candidate_df["visual_score"] = visual_scores

    # hybrid rerank ONLY for dataset images
    text_query = text_embeddings[image_index].reshape(1,-1)
    semantic_scores = cosine_similarity(text_query, text_embeddings[candidate_df.index])[0]

    candidate_df["hybrid"] = 0.6*visual_scores + 0.4*semantic_scores #compares visual and txtual information and assigns weightage to both.

    ranked = candidate_df.sort_values("hybrid", ascending=False)
    ranked = ranked[ranked.index != image_index]

    return ranked.head(top_k)

# ================= UPLOADED IMAGE SEARCH (VISUAL ONLY) =================
def retrieve_from_uploaded(image_path, top_k=3):

    query_vec = extract_query_embedding(image_path).reshape(1,-1) #converts 1D array of the image uploaded to a 2D array
    scores = cosine_similarity(query_vec, image_embeddings)[0] #[0] again converts to 1D array

    df_copy = df.copy()
    df_copy["similarity"] = scores
    ranked = df_copy.sort_values("similarity", ascending=False).head(top_k)

    results = []
    for _, row in ranked.iterrows():
        results.append((row["image_filename"], float(row["similarity"])))

    return results

# ================= HELPERS =================
def get_image_path(food_name):
    match = df[df["food_name"].str.lower() == food_name.lower()]
    if len(match)==0:
        return None
    return str(ASSET_DIR / match.iloc[0]["image_filename"])


# ================= DEBUG TEST =================
if __name__ == "__main__":
    test_img = ASSET_DIR / df.iloc[0]["image_filename"]

    print("\nTesting embedding retrieval on:", test_img)

    res = retrieve_from_uploaded(test_img, top_k=5)

    print(res[["food_name", "similarity"]])
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd  
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" 

# load embeddings
food_embeddings = np.load(MODEL_DIR / "dino_embeddings.npy")

# load metadata
df = pd.read_csv(DATA_DIR / "food_metadata.csv")  

# SAME MODEL AS EMBEDDING CREATION
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
model.eval()


def extract_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]  # CLS token

    return emb.cpu().numpy()

def detect_veg_hint(image_path):
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).resize((64, 64))
    arr = np.array(img)

    red = arr[:, :, 0].mean()
    green = arr[:, :, 1].mean()
    blue = arr[:, :, 2].mean()

    veg_score = green + blue
    nonveg_score = red

    if veg_score > nonveg_score * 1.05:
        return "vegetarian"

    return None

def search_similar(image_path, top_k=3):

    # -------------------------
    # Step 1: Visual similarity
    # -------------------------
    query_emb = extract_embedding(image_path)
    visual_scores = cosine_similarity(query_emb, food_embeddings)[0]

    # -------------------------
    # Step 2: veg detection FIRST
    # -------------------------
    veg_hint = detect_veg_hint(image_path)

    if veg_hint == "vegetarian":
        detected_diet = "vegetarian"
    else:
        top_k_indices = np.argsort(visual_scores)[::-1][:5]
        diet_votes = [str(df.iloc[idx]["veg_nonveg"]).lower() for idx in top_k_indices]

        # normalize votes
        diet_votes = ["vegetarian" if "veg" in d and "non" not in d else "meat" for d in diet_votes]

        detected_diet = Counter(diet_votes).most_common(1)[0][0]

    print("FINAL DIET:", detected_diet)

    # -------------------------
    # Step 3: Semantic similarity
    # -------------------------
    from modules.text import embeddings as text_embeddings

    top_visual_idx = np.argmax(visual_scores)

    text_query = text_embeddings[top_visual_idx].reshape(1, -1)
    semantic_scores = cosine_similarity(text_query, text_embeddings)[0]

    # -------------------------
    # Step 4: Hybrid scoring
    # -------------------------
    hybrid_scores = 0.6 * visual_scores + 0.4 * semantic_scores

    # -------------------------
    # Step 5: HARD FILTER
    # -------------------------
    filtered_indices = []
    filtered_scores = []

    for i, score in enumerate(hybrid_scores):

        row_diet = str(df.iloc[i]["veg_nonveg"]).lower().strip()

        is_veg = ("veg" in row_diet and "non" not in row_diet)

        if detected_diet == "vegetarian":
            if not is_veg:
                continue
        else:
            if is_veg:
                continue

        filtered_indices.append(i)
        filtered_scores.append(score)

    if len(filtered_indices) == 0:
        return None, None

    filtered_indices = np.array(filtered_indices)
    filtered_scores = np.array(filtered_scores)

    # -------------------------
    # Step 6: Top-k selection
    # -------------------------
    sorted_idx = np.argsort(filtered_scores)[::-1][:top_k]

    top_idx = filtered_indices[sorted_idx]
    top_scores = filtered_scores[sorted_idx]

    if top_scores[0] < 0.35:
        return None, None

    return top_idx, top_scores
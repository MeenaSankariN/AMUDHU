#This code file is the decision rules code file for text amd speech modalities. The core backbone of the system's retrieval intelligence from the raw meta data
#This code has a hybrid decision making algorthm- cosine similarity + score/weightage to features.
#This code receives the input query from the user, converts to minilm embeddings and then proceeds with the decision making process
import os
import numpy as np
import pandas as pd
from pathlib import Path
from rapidfuzz import process
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =========================
# Load resources
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

df = pd.read_csv(DATA_DIR / "food_metadata.csv") #meta data
embeddings = np.load(MODEL_DIR / "minilm_embeddings.npy") #existing embedding created in offline

model = SentenceTransformer("all-MiniLM-L6-v2") 
#sentence transformer- converts to 384 dim vector. Has 6 layers of tranformer blocks each with 12 attention heads.

# normalize columns- for cleaner data in the meta dataset
df["state"] = df["state"].str.strip()
df["region"] = df["region"].str.strip()
df["veg_nonveg"] = df["veg_nonveg"].str.lower().str.strip()
df["taste"] = df["taste"].str.lower().str.strip()
df["course_type"] = df["course_type"].str.lower().str.strip()

# =========================
# helpers
# =========================

VALID_REGIONS = [
    "north india", "south india", "east india", "west india",
    "north-east india", "northeast india", "union territories",
    "central india", "pondicherry"
]

#this is for the query side. Although now it is not needed because of limited user input options, it will be needed in further extension

def normalize_text(text):
    if text is None:
        return None
    return text.lower().replace(" ", "").replace("-", "").strip()

#rapid fuzzy function- it uses string similarity and accepts two words with similar spelling as the same word for a threshold value
def fuzzy_match(user_input, options, threshold=80):
    if not user_input:
        return None
    result = process.extractOne(user_input, options)
    if result and result[1] >= threshold:
        return result[0]
    return None

# =========================
# location validation- This is for location check and acceptance
# =========================

def validate_location(user_input):
    if not user_input:
        return None, None

    user_norm = normalize_text(user_input)

    states = df["state"].unique().tolist()
    state_norm = [normalize_text(s) for s in states]
    region_norm = [normalize_text(r) for r in VALID_REGIONS]

    state_match = process.extractOne(user_norm, state_norm)
    region_match = process.extractOne(user_norm, region_norm)

    if state_match and state_match[1] >= 80:
        return "state", states[state_match[2]]

    if region_match and region_match[1] >= 80:
        return "region", VALID_REGIONS[region_match[2]]

    return None, None

# =========================
# semantic ranking- cosine similarity + boosting weightage for taste, diet and location
# =========================

def semantic_rank(df_subset, query, taste=None, diet=None, location=None): #scores are calculated based on each feature of the food and query.

    query_vec = model.encode([query])
    subset_embeddings = embeddings[df_subset.index]

    semantic_scores = cosine_similarity(query_vec, subset_embeddings)[0]

    ranked = df_subset.copy()
    ranked["semantic_score"] = semantic_scores
    ranked["score"] = semantic_scores

    # -------------------------
    # Taste boost
    # -------------------------
    if taste:
        ranked.loc[ranked["taste"] == taste, "score"] += 0.10

    # -------------------------
    # Diet boost
    # -------------------------
    if diet:
        ranked.loc[ranked["veg_nonveg"] == diet, "score"] += 0.10

    # -------------------------
    # Location boost
    # -------------------------
    if location:
        ranked.loc[ranked["state"] == location, "score"] += 0.05
        ranked.loc[ranked["region"].str.lower() == location.lower(), "score"] += 0.03

    return ranked.sort_values("score", ascending=False)

# =========================
# MAIN ENGINE- rule based decision making- this is the intelligence of this retrieval system
# =========================

def recommend_food(location=None, course=None, taste=None, diet=None, speech_region=None):

    reason_base = []

    # --- location handling ---------------------------
    if speech_region:  # speech mode
        if speech_region in df["state"].values:
            loc_type, loc_value = "state", speech_region
            reason_base.append("speech_inferred_state")
        else:
            loc_type, loc_value = "region", speech_region
            reason_base.append("speech_inferred_region")

    else:  # normal text mode
        loc_type, loc_value = validate_location(location)
        if not loc_value:
            return {"status": "error", "reason": "invalid_location"}

    # --- normalize course ---
    valid_courses = ["main course", "snack", "dessert", "beverage"]
    if course not in valid_courses:
        course = "main course"

    # --- build a coherent sentence for the LM to learn-------
    query = f"{course} food from {loc_value}"
    if taste:
        query += f" that is {taste}"
    if diet:
        query += f" and {diet}"

    # --- filtering ---
    mask = (df["course_type"] == course)

    if diet:
        mask &= (df["veg_nonveg"] == diet)

    df_filtered = df[mask]

    # --- location subset ---
    if loc_type == "state":
        local_subset = df_filtered[df_filtered["state"] == loc_value]
        reason_base.append("matched_state")

    else:
        local_subset = df_filtered[df_filtered["region"].str.lower() == loc_value.lower()]

        # remove duplicate dishes across states
        local_subset = local_subset.drop_duplicates(subset=["food_name"])

        reason_base.append("matched_region")

    results = None
    fallback_tag = None

    # ------------------------------------------------
    # Case 1: local match
    # ------------------------------------------------
    if len(local_subset) >= 1:

        ranked = semantic_rank(
            local_subset,
            query,
            taste=taste,
            diet=diet,
            location=loc_value
        ).head(12)

        if len(ranked) >= 3:
            results = ranked.sample(n=3, weights="score", replace=False) #considers only top 3 foods, others are left
        else:
            results = ranked

        reason_base.append("semantic_match")# this is later used in rag explainer for explaining the mistral model

    # ------------------------------------------------
    # Case 2: same course anywhere
    # ------------------------------------------------
    elif len(df_filtered) >= 1:

        ranked = semantic_rank(
            df_filtered,
            query,
            taste=taste,
            diet=diet,
            location=loc_value
        ).head(12)

        if len(ranked) >= 3:
            results = ranked.sample(n=3, weights="score", replace=False)
        else:
            results = ranked

        fallback_tag = "fallback_same_course"

    # ------------------------------------------------
    # Case 3: snack/dessert swap
    # ------------------------------------------------
    elif course in ["snack", "dessert"]:

        alt = "dessert" if course == "snack" else "snack"
        alt_subset = df[df["course_type"] == alt]

        ranked = semantic_rank(
            local_subset,
            query,
            taste=taste,
            diet=diet,
            location=loc_value
        ).head(12)

        if len(ranked) >= 3:
            results = ranked.sample(n=3, weights="score", replace=False)
        else:
            results = ranked

        fallback_tag = "fallback_course_swap"

    # ------------------------------------------------
    # Case 4: beverage fallback
    # ------------------------------------------------
    elif course == "beverage":

        bev_subset = df[df["course_type"] == "beverage"]

        if loc_type == "region":
            bev_subset = bev_subset.drop_duplicates(subset=["food_name"])

        ranked = semantic_rank(
            local_subset,
            query,
            taste=taste,
            diet=diet,
            location=loc_value
        ).head(12)

        if len(ranked) >= 3:
            results = ranked.sample(n=3, weights="score", replace=False)
        else:
            results = ranked

        fallback_tag = "fallback_beverage"

    # ------------------------------------------------
    # Case 5: diverse fallback
    # ------------------------------------------------
    else:

        diverse = pd.concat([
            df[df["course_type"] == "dessert"],
            df[df["course_type"] == "beverage"],
            df[df["veg_nonveg"] == "vegetarian"],
            df[df["veg_nonveg"] == "meat-based"],
            df[df["course_type"] == "snack"]
        ]).drop_duplicates()

        results = diverse.sample(n=min(3, len(diverse)))

        fallback_tag = "fallback_diverse"

    # ------------------------------------------------
    # build output
    # ------------------------------------------------
    output = []

    for _, row in results.iterrows():

        tags = reason_base.copy()

        if fallback_tag:
            tags.append(fallback_tag)

        output.append({
            "food_name": row["food_name"],
            "state": row["state"],
            "region": row["region"],
            "course_type": row["course_type"],
            "veg_nonveg": row["veg_nonveg"],
            "taste": row.get("taste", None),
            "description": row["description"],
            "score": float(row.get("score", 0.0)),
            "image_filename": row["image_filename"],
            "reason_tags": tags
        })

    return {"status": "ok", "results": output}
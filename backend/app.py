from logger import logger
import time
import uuid
import os
import pandas as pd

from modules.speech_parser import parse_preferences
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from modules.security import sanitize_text
from modules.decision_engine import recommend_food
from modules.speech import predict_language
from modules.image_infer import search_similar

from rag.explainer import generate_explanation
from rag.llm_rewriter import rewrite_explanation
from rag.system_message import generate_system_message


# =========================
# CONFIG
# =========================
MAX_IMAGE_SIZE = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]

df = pd.read_csv("data/food_metadata.csv")

app = FastAPI(title="AMUDHU Food Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/food_images", StaticFiles(directory="assets/food_images"), name="food_images")


ACCENT_TO_STATES = {
    "tamil": ["Tamil Nadu"],
    "malayalam": ["Kerala"],
    "kannada": ["Karnataka"],
    "telugu": ["Andhra Pradesh", "Telangana"]
}


# =========================
# REQUEST SCHEMAS
# =========================
class TextRequest(BaseModel):
    location: str | None = None
    course: str | None = None
    taste: str | None = None
    diet: str | None = None


class SpeechConfirmRequest(BaseModel):
    region: str
    course: str = "main course"
    taste: str | None = None
    diet: str | None = None


class SpeechPreferenceRequest(BaseModel):
    region: str
    transcript: str


# =========================
# SAFE WRAPPER
# =========================
def safe_recommend(**kwargs):
    try:
        response = recommend_food(**kwargs)

        if response["status"] != "ok":
            return {
                "status": "fallback",
                "results": response.get("results", [])
            }

        return response

    except Exception as e:
        logger.error(f"[ERROR] {e}")
        return {
            "status": "error",
            "message": "Something went wrong. Please try again."
        }


# =========================
# TEXT
# =========================
@app.post("/recommend/text")
def recommend_text(req: TextRequest):

    location = sanitize_text(req.location)
    course = sanitize_text(req.course)
    taste = sanitize_text(req.taste)
    diet = sanitize_text(req.diet)

    response = safe_recommend(
        location=location,
        course=course,
        taste=taste,
        diet=diet
    )

    if response["status"] == "error":
        return response

    results = []

    for item in response.get("results", []):
        try:
            raw_explanation = {
                "food_name": item["food_name"],
                "state": item["state"],
                "region": item["region"],
                "course_type": item["course_type"],
                "taste": item["taste"]
            }

            explanation = rewrite_explanation({
                **raw_explanation,
                "explanation": generate_explanation(raw_explanation)
            })

        except Exception as e:
            logger.error(f"[TEXT] Explanation error: {e}")
            explanation = "This dish matches your preferences."

        results.append({
            "name": item["food_name"],
            "taste": item["taste"],
            "course_type": item["course_type"],
            "diet": item["veg_nonveg"],
            "state": item["state"],
            "region": item["region"],
            "image": f"/food_images/{item['image_filename']}",
            "score": float(item.get("score", 0)),
            "reason": item.get("reason_tags", []),
            "about": explanation
        })

    system_message = generate_system_message(
        input_type="text",
        preferences=req.dict(),
        fallback = any(
            "fallback" in reason
            for item in response.get("results", [])
            for reason in item.get("reason_tags", [])
        )
    )

    return {
        "status": response["status"],
        "input_type": "text",
        "system_message": system_message,
        "results": results
    }


# =========================
# IMAGE
# =========================
@app.post("/recommend/image")
async def recommend_from_image(file: UploadFile = File(...)):

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return {"status": "error", "message": "Unsupported image format"}

    contents = await file.read()

    if len(contents) > MAX_IMAGE_SIZE:
        return {"status": "error", "message": "Image too large"}

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(contents)

    idxs, scores = search_similar(temp_path)
    os.remove(temp_path)

    if idxs is None:
        return {
            "status": "fallback",
            "system_message": "I couldn’t confidently recognize this dish. Try another image.",
            "results": []
        }

    results = []

    for i, score in zip(idxs, scores):
        row = df.iloc[i]

        try:
            raw = {
                "food_name": row["food_name"],
                "state": row["state"],
                "region": row["region"],
                "course_type": row["course_type"],
                "taste": row["taste"]
            }

            explanation = rewrite_explanation({
                **raw,
                "explanation": generate_explanation(raw)
            })

        except:
            explanation = "This dish looks similar to your uploaded image."

        results.append({
            "name": row["food_name"],
            "state": row["state"],
            "region": row["region"],
            "taste": row["taste"],
            "course_type": row["course_type"],
            "diet": row["veg_nonveg"],
            "image": f"/food_images/{row['image_filename']}",
            "score": round(float(score), 4),
            "reason": ["image_similarity"],
            "about": explanation
        })

    system_message = generate_system_message(
        input_type="image",
        inferred={
            "foods": [r["name"] for r in results]
        }
    )

    return {
        "status": "ok",
        "input_type": "image",
        "system_message": system_message,
        "results": results
    }


# =========================
# SPEECH DETECTION
# =========================
@app.post("/recommend/speech")
async def speech_detect(audio: UploadFile = File(...)):

    temp_path = f"temp_{audio.filename}"

    with open(temp_path, "wb") as f:
        f.write(await audio.read())

    prediction = predict_language(temp_path)
    os.remove(temp_path)

    if not isinstance(prediction, dict):
        return {"stage": "error", "message": "Speech prediction failed."}

    accent = prediction.get("language")
    confidence = prediction.get("confidence")

    if confidence is None or confidence < 0.25:
        return {
            "stage": "low_confidence",
            "message": "I couldn’t understand your accent clearly."
        }

    states = ACCENT_TO_STATES.get(accent.lower())

    if not states:
        return {"stage": "error", "message": "Accent not supported"}

    if len(states) > 1:
        return {
            "stage": "choose_state",
            "options": states,
            "system_message": generate_system_message(
                input_type="speech_confirm",
                inferred={"region": accent.title()}
            )
        }

    return {
        "stage": "confirm_region",
        "predicted_region": states[0],
        "system_message": generate_system_message(
            input_type="speech_confirm",
            inferred={"region": states[0]}
        )
    }


# =========================
# SPEECH CONFIRM
# =========================
@app.post("/recommend/speech_confirmed")
def recommend_from_speech(req: SpeechConfirmRequest):

    response = safe_recommend(
        location=req.region,
        course=req.course,
        taste=req.taste,
        diet=req.diet
    )

    results = []

    for item in response.get("results", []):
        explanation = generate_explanation(item)

        results.append({
            "name": item["food_name"],
            "state": item["state"],
            "region": item["region"],
            "taste": item["taste"],
            "course_type": item["course_type"],
            "diet": item["veg_nonveg"],
            "image": f"/food_images/{item['image_filename']}",
            "score": float(item.get("score", 0)),
            "reason": item.get("reason_tags", []),
            "about": explanation
        })

    return {
        "status": response["status"],
        "input_type": "speech",
        "system_message": generate_system_message(
            input_type="speech",
            preferences={"region": req.region},
            fallback = any(
            "fallback" in reason
            for item in response.get("results", [])
            for reason in item.get("reason_tags", [])
        )
        ),
        "results": results
    }


# =========================
# SPEECH PREFERENCES
# =========================
@app.post("/recommend/speech_preferences")
def recommend_from_speech_text(req: SpeechPreferenceRequest):

    safe_text = sanitize_text(req.transcript)

    taste, course, diet = parse_preferences(safe_text)

    # -------------------------
    # Normalize inputs
    # -------------------------
    if not course:
        course = "main course"

    if course != "main course":
        taste = None
        diet = None

    # -------------------------
    # Recommendation
    # -------------------------
    response = safe_recommend(
        location=req.region,
        course=course,
        taste=taste,
        diet=diet
    )

    results = []

    for item in response.get("results", []):

        try:
            #
            raw_explanation = {
                "food": item["food_name"],
                "state": item["state"],
                "region": item["region"],
                "course": item["course_type"],
                "taste": item["taste"],
                "explanation": generate_explanation(item)
            }

            explanation = rewrite_explanation(raw_explanation)

        except Exception as e:
            logger.error(f"[SPEECH] Explanation error: {e}")
            explanation = "This dish matches your preferences."

        results.append({
            "name": item["food_name"],
            "state": item["state"],
            "region": item["region"],
            "taste": item["taste"],
            "course_type": item["course_type"],
            "diet": item["veg_nonveg"],
            "image": f"/food_images/{item['image_filename']}",
            "score": float(item.get("score", 0)),
            "reason": item.get("reason_tags", []),
            "about": explanation   # ✅ matches frontend
        })

    # -------------------------
    # Detect fallback properly
    # -------------------------
    fallback = any(
        "fallback" in reason
        for item in response.get("results", [])
        for reason in item.get("reason_tags", [])
    )

    # -------------------------
    # System message
    # -------------------------
    system_message = generate_system_message(
        input_type="speech",
        preferences={
            "region": req.region,
            "course": course,
            "taste": taste,
            "diet": diet
        },
        fallback=fallback
    )

    return {
        "status": response.get("status", "ok"),
        "input_type": "speech",
        "system_message": system_message,
        "results": results
    }
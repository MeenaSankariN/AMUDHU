from modules.speech import predict_language
from modules.decision_engine import recommend_food
from modules.image import find_similar_images, get_index_by_food, get_image_path


# ===============================
# IMAGE RETRIEVAL
# ===============================
def attach_visuals(result):

    if result["status"] != "ok":
        return result

    for item in result["results"]:

        # primary image (ground truth)
        item["image"] = get_image_path(item["food_name"])

        # supporting visual evidence
        idx = get_index_by_food(item["food_name"])

        if idx is None:
            item["visual_variants"] = []
            continue

        visuals = find_similar_images(
            idx,
            top_k=3,
            region=item["region"],
            diet=item["veg_nonveg"],
            course=item["course_type"]
        )

        item["visual_variants"] = visuals["food_name"].tolist()

    return result


# ===============================
# TEXT MODE
# ===============================
def run_text_mode(location, course, taste=None, diet=None):

    result = recommend_food(
        location=location,
        course=course,
        taste=taste,
        diet=diet
    )

    # IMPORTANT: attach images
    result = attach_visuals(result)

    return {
        "mode": "text",
        "data": result
    }


# ===============================
# SPEECH MODE
# ===============================
def run_speech_mode(audio_path, course, taste=None, diet=None, confirm_region=True):

    speech_result = predict_language(audio_path)

    inferred_region = speech_result["language"]
    confidence = speech_result["confidence"]

    region_map = {
        "Tamil": "South India",
        "Telugu": "South India",
        "Kannada": "South India",
        "Malayalam": "South India"
    }

    suggested_region = region_map.get(inferred_region, None)

    if confirm_region is False:
        return {
            "mode": "speech",
            "status": "need_manual_location",
            "accent_guess": inferred_region,
            "confidence": confidence
        }

    result = recommend_food(
        speech_region=suggested_region,
        course=course,
        taste=taste,
        diet=diet
    )

    result = attach_visuals(result)

    return {
        "mode": "speech",
        "accent_detected": inferred_region,
        "confidence": confidence,
        "data": result
    }
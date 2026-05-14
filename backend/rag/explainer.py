#this file is for the explanation of why the food is chosen. This builds a raw explanation to the mistral model to rephrase the output and its reasoning.
#this file includes fallback conditions, clean chunks for the selected foods to create a raw explanation

from rag.retriever import retrieve_food_context
from rag.llm_rewriter import rewrite_explanation


# ------------------------------------------------
# REASON TAG EXPLANATION
# ------------------------------------------------
def explain_reason_tags(tags): 

    if "fallback_same_course" in tags:
        return "no food from your location matched, so a similar course was suggested"

    if "fallback_course_swap" in tags:
        return "no exact match was found, so a similar type of dish was suggested"

    if "fallback_beverage" in tags:
        return "no suitable food was found, so a beverage alternative was suggested"

    if "fallback_diverse" in tags:
        return "no close match was found, so a diverse recommendation was provided"

    reasons = []

    if "matched_state" in tags:
        reasons.append("it comes from your requested state")

    elif "matched_region" in tags:
        reasons.append("it belongs to your requested region")

    if "semantic_match" in tags:
        reasons.append("it matches your taste preference")

    if "speech_inferred_region" in tags:
        reasons.append("your accent suggested this cuisine")

    if not reasons:
        return "it matches your preferences"

    return " and ".join(reasons)


# ------------------------------------------------
# CLEAN KNOWLEDGE CHUNK- converts into almost natural language
# ------------------------------------------------
def clean_fact(food, text):

    text = text.replace(food, "").strip(": ")

    starters = [
        "is a ", "is an ", "is ",
        "has a ", "has an ", "has "
    ]

    for s in starters:
        if text.lower().startswith(s):
            text = text[len(s):]

    replacements = {
        "sweet taste": "sweet in taste",
        "savoury taste": "savoury in taste",
        "spicy taste": "spicy in taste",
        "tangy taste": "tangy in taste",
        "mild taste": "mild in taste"
    }

    lower_text = text.lower() 

    for k, v in replacements.items(): #replace the taste words
        if k in lower_text:
            text = v

    text = text.strip(". ")  #removes full stops

    if len(text) > 0: #captalize first word
        text = text[0].upper() + text[1:]

    return text


# ------------------------------------------------
# MAIN EXPLANATION FUNCTION
# ------------------------------------------------
def generate_explanation(food_item):

    food = food_item.get("food_name") or food_item.get("food")
    state = food_item.get("state")
    region = food_item.get("region")
    course = food_item.get("course_type")
    taste = food_item.get("taste")
    tags = food_item.get("reason_tags", [])

    # -------------------------------
    # BASE REASONING
    # -------------------------------
    logic = explain_reason_tags(tags) #explains why this/these foods where chosen

    explanation = f"{food} was recommended because {logic}."

    # -------------------------------
    # STRUCTURED DETAILS FOR EXPLANATION
    # -------------------------------
    details = []

    if course:
        details.append(course)

    if taste:
        details.append(f"{taste} in taste")

    if state:
        details.append(f"from {state}")

    if len(details) == 1:
        explanation += f" It is a {details[0]}."

    elif len(details) > 1:
        explanation += " It is " + ", ".join(details[:-1]) + " and " + details[-1] + "."

    # -------------------------------
    # RETRIEVE RAG FACTS
    # -------------------------------
    facts = retrieve_food_context(food, tags, k=10)

    context_sentences = []

    if facts:
        for f in facts:

            cleaned = clean_fact(food, f)

            if cleaned:
                context_sentences.append(cleaned)

            if len(context_sentences) >= 2: #only two sentences
                break

    # -------------------------------
    # BUILD DESCRIPTION FROM RAG
    # -------------------------------
    if context_sentences:
        description = " ".join(context_sentences)

    else:
        description = f"{food} is a traditional dish from {state} in {region}."

    description = description[:320].strip()

    if not description.lower().startswith(food.lower()):
        description = f"{food} is {description[0].lower() + description[1:]}"

    return description
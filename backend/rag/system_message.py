
def generate_system_message(input_type, preferences=None, inferred=None, fallback= False):
    """
    Creates a user-facing explanation of what the system understood
    """

    preferences = preferences or {}
    inferred = inferred or {}
    if preferences.get("course") != "main course":
        preferences.pop("taste", None)
        preferences.pop("diet", None)
        
    if input_type == "text":
        parts = []

        if preferences.get("taste"):
            parts.append(preferences["taste"])

        if preferences.get("diet"):
            parts.append(preferences["diet"])

        if preferences.get("course"):
            parts.append(preferences["course"])

        if preferences.get("location"):
            parts.append(f"from {preferences['location']}")

        description = " ".join(parts)
        
        if fallback:
            return f"You asked for {description}, but I couldn't find an exact match. Here are some similar dishes you might enjoy."

        return f"You asked for {description}. Here are(is) the best match(es)."


    elif input_type == "speech":
        
        region = preferences.get("region")
        course = preferences.get("course")
        
        if fallback:
            return f"I couldn’t find an exact {course} match from {region}, but here are some similar dishes you might like."

        return f"Here are some {course} dishes from {region} that match your preference."

        
    elif input_type == "speech_confirm":
        region = inferred.get("region")

        return f"Your speech sounds closest to {region} English. Would you like me to explore dishes from {region} cuisine?"


    elif input_type == "speech_selected":
        region = inferred.get("region")

        return f"Great! I will recommend dishes from {region} cuisine."


    elif input_type == "image":

        foods = inferred.get("foods")

        if isinstance(foods, list) and len(foods) > 0:
            top_food = foods[0]
            return f"The uploaded dish looks similar to {top_food}. Here are visually similar dishes you may enjoy."

        if fallback:
            return "I couldn’t confidently identify the exact dish, but here are some visually similar options."

        return "Here are dishes that look similar to your uploaded image."


    return ""
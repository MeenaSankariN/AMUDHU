ACCENT_TO_STATE = {
    "tamil": "Tamil Nadu",
    "telugu": "Andhra Pradesh",
    "kannada": "Karnataka",
    "malayalam": "Kerala"
}

def accent_to_region(label):
    if not label:
        return None
    label = label.lower()
    return ACCENT_TO_STATE.get(label)
TASTE_WORDS = ["spicy", "tangy", "sweet", "savory"]
COURSE_WORDS = ["main course", "snack", "dessert", "beverage"]
DIET_WORDS = ["vegetarian", "meat-based"]


def parse_preferences(text: str):

    if not text:
        return None, None, None

    text = text.lower()

    taste = None
    course = None
    diet = None

    for t in TASTE_WORDS:
        if t in text:
            taste = t
            break

    for c in COURSE_WORDS:
        if c in text:
            course = c
            break

    for d in DIET_WORDS:
        if d in text:
            diet = d
            break

    return taste, course, diet
from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners import Toxicity

prompt_scanner = PromptInjection()
toxicity_scanner = Toxicity()

MAX_LENGTH = 500


def sanitize_text(text: str):

    if text is None:
        return text

    if len(text) > MAX_LENGTH:
        raise ValueError("Input too long")

    cleaned, valid, risk = prompt_scanner.scan(text)

    if not valid:
        raise ValueError("Prompt injection detected")

    cleaned, valid, risk = toxicity_scanner.scan(cleaned)

    return cleaned
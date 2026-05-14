#this file is for the decoder or the text output generator where mistral-7b is used as a gpt- generated unified formatted file available from Hugging face.
#this is version 2 with 7 billion parameters. This model is specifically designed for question answering, chatbot tasks.
#Pre-quantized. This model is 4 bit and K quantized.
#All the weights will be stored in 0-15 (0000 to 1111) format. So to prevent the loss of decimal points while compressing to 4 bit, this model uses K quantization.
#K-quan.: Group the weights, scale them for being under the value-15 and use them.
#M stands for medium variant- medium in speed and memory

from llama_cpp import Llama
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048, #context size- max of 2048 token can be handled at a time
    n_threads=8, #number of cores of cpu
    n_gpu_layers=0 #no gpu needed
)


# ---------------------------------------------------
# LLM REWRITER
# ---------------------------------------------------

def rewrite_explanation(explanation):

    prompt = f"""
Rewrite the following food description into a clear and natural sentence.

Description:
{explanation}

Rules:
- Write 1–2 sentences
- Do not repeat words
- Do not mention user preferences
- Only describe the food

Final description:
"""

    output = llm(
        prompt, #the input text
        max_tokens=120, #should be less than or equal to 120
        temperature=0.5, #chooses balance between random/creative words and deterministic (given food description) words
        top_p=0.9 #chooses the words from almost all the probable words related to the food
    )

    response = output["choices"][0]["text"].strip()

    if "Final description:" in response:
        response = response.split("Final description:")[-1].strip() #remove the "final description:"

    return response


# ---------------------------------------------------
# TEXT MODALITY EXPLANATION
# ---------------------------------------------------

def build_text_explanation(food_type, taste, region, dish):

    explanation = f"""
The user prefers {food_type} food with a {taste} taste profile from {region}.
{dish} matches these preferences and is recommended.
"""

    return rewrite_explanation(explanation)


# ---------------------------------------------------
# SPEECH MODALITY EXPLANATION
# ---------------------------------------------------

def build_speech_explanation(accent, food_type, taste, region, dish):

    explanation = f"""
The speech model detected a {accent} accent, suggesting preference for cuisine from that region.
The user prefers {food_type} food with a {taste} taste profile from {region}.
{dish} satisfies these preferences and is recommended.
"""

    return rewrite_explanation(explanation)


# ---------------------------------------------------
# IMAGE MODALITY EXPLANATION
# ---------------------------------------------------

def build_image_explanation(retrieved_foods):

    explanation = """
The system analyzed the uploaded food image and compared it with dishes in the dataset using visual embeddings.
"""

    for food, score in retrieved_foods:
        explanation += f"\n{food} was found with similarity score {score:.2f}."

    best_food = retrieved_foods[0][0]

    explanation += f"""
{best_food} has the highest visual similarity and is selected as the final recommendation.
"""

    return rewrite_explanation(explanation)
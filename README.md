
# AMUDHU 🍛

## Multimodal Indian Cuisine Recommendation System using Artificial Intelligence

AMUDHU is a multimodal AI-powered Indian cuisine recommendation system designed to recommend culturally relevant Indian food items using **text, speech, and image inputs**. The system combines machine learning, deep learning, retrieval-augmented generation (RAG), and lightweight LLM-based explanation generation to provide personalized and explainable food recommendations.

The project is built on a curated dataset containing more than **350 Indian food items** from **26 Indian states and 8 Union Territories**.

---

# ✨ Features

## 🔤 Text-based Food Recommendation

* Takes natural language food preference queries
* Uses semantic embeddings for similarity matching using MiniLM L6 V2
* Returns top-k culturally relevant food recommendations

## 🎤 Speech-based Recommendation

* Detects South Indian English accents
* Supports:

  * Tamil
  * Telugu
  * Kannada
  * Malayalam
* Uses MFCC feature extraction with machine learning classification
* Converts speech to text for downstream recommendation

## 🖼️ Image-based Food Retrieval

* Accepts food images as input
* Uses image embeddings for similarity retrieval using DINO v2 small
* Recommends visually similar Indian dishes

## 🧠 Decision Engine

* Multimodal ranking and rule-based filtering
* Combines:

  * cosine similarity
  * preference weighting
  * contextual filtering
  * fallback handling

## 📚 Retrieval-Augmented Generation (RAG)

* Uses FAISS-based retrieval
* Retrieves food metadata and explanation chunks
* Generates explainable recommendations

## 🤖 LLM-based Response Rephrasing

* Uses Mistral 7B for conversational explanation generation
* Converts raw outputs into user-friendly recommendations

## 🛡️ Guardrails and Validation

* Toxicity filtering
* Input validation checks
* Context-aware handling

# 🧰 Technology Stack

## Backend

* FastAPI
* Python
* FAISS
* NumPy
* Pandas
* Scikit-learn
* Sentence Transformers
* Librosa
* OpenCV

## Frontend

* React.js
* HTML
* CSS
* JavaScript

## Machine Learning / Deep Learning

* MFCC Feature Extraction
* XGBoost
* Cosine Similarity Retrieval
* DINOv2 Embeddings
* MiniLM Embeddings
* Retrieval-Augmented Generation (RAG)

## LLM

* Mistral 7B Instruct

---

# 📂 Project Structure

```text
AMUDHU/
│
├── Architecture/
├── backend/
│   ├── assets/
│   ├── data/
│   ├── models/
│   ├── modules/
│   ├── pipeline/
│   ├── rag/
│   ├── app.py
│   └── logger.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
│
├── UI_screenshot/
├── README.md
└── .gitignore
```

---

# ⚙️ Backend Workflow

## Text Pipeline

1. User enters food preference query
2. FastAPI receives request
3. Text embedding generated using MiniLM
4. Decision engine performs semantic similarity ranking
5. Top-k recommendations retrieved
6. RAG retrieves contextual explanation chunks
7. Mistral LLM rewrites response naturally

---

## Speech Pipeline

1. User uploads audio
2. Audio preprocessing performed
3. MFCC features extracted
4. Accent classification using XGBoost
5. Speech converted to text
6. Text passed to decision engine
7. Final recommendations generated

---

## Image Pipeline

1. User uploads food image
2. DINOv2 embeddings generated
3. Cosine similarity performed against dataset embeddings
4. Similar food items retrieved
5. Recommendations returned to user

---

# 🧠 Decision Engine Summary

The decision engine is the core recommendation module of the system.

It combines:

* semantic similarity scoring
* cosine similarity ranking
* preference weighting
* contextual filtering
* fallback handling

The engine processes embeddings generated from different modalities and produces the top ranked Indian food recommendations.

---

# 📚 Retrieval-Augmented Generation (RAG)

The RAG layer is used only for explanation generation.

It:

* retrieves relevant food knowledge chunks
* accesses metadata from the FAISS index
* provides contextual information to the LLM
* improves explainability of recommendations

This design keeps the recommendation engine lightweight while enhancing user interaction quality.

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/MeenaSankariN/AMUDHU.git
```

---

# 🔧 Backend Setup

## Create Virtual Environment

```bash
python -m venv amudhu
```

## Activate Environment

### Windows

```bash
amudhu\Scripts\activate
```

### Linux / Mac

```bash
source amudhu/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Backend

```bash
uvicorn app:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

---

# 💻 Frontend Setup

Navigate to frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Run frontend:

```bash
npm start
```

Frontend runs at:

```text
http://localhost:3000
```
---

# 🎓 Academic Contribution

This project demonstrates:

* multimodal AI integration
* explainable recommendation systems
* retrieval-augmented generation
* speech processing
* image retrieval systems
* full-stack AI application development

---

# 👨‍💻 Author

## Meena Sankari

Master's Student – Data Science

GitHub:
https://github.com/MeenaSankariN

---


This project is intended for academic and research purposes.


from pathlib import Path
import numpy as np
import librosa
import joblib
import tensorflow as tf
import tensorflow_hub as hub

# =========================
# Load trained artifacts
# =========================

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

model = joblib.load(MODEL_DIR / "language_xgb.pkl")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

# =========================
# Config
# =========================
TARGET_SR = 16000
DURATION = 4
SAMPLES = TARGET_SR * DURATION
N_MFCC = 40

# =========================
# Audio preprocessing
# =========================
def load_audio(path):

    y, sr = librosa.load(path, sr=16000)
    if np.mean(np.abs(y)) < 0.005:
        return None
    y, _ = librosa.effects.trim(y, top_db=35)

    # enforce minimum length
    min_len = 16000 * 2

    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))

    return y


# =========================
# Feature extraction
# =========================
def extract_features(y):

    y = y.astype(np.float32)

    scores, embeddings, spectrogram = yamnet_model(y)

    # Detect if audio contains speech
    speech_score = tf.reduce_mean(scores[:, 0:4])  # speech related classes

    if speech_score < 0.15:
        return None

    mean = tf.reduce_mean(embeddings, axis=0)
    std = tf.math.reduce_std(embeddings, axis=0)
    maxv = tf.reduce_max(embeddings, axis=0)

    if embeddings.shape[0] > 1:
        delta = embeddings[1:] - embeddings[:-1]
        delta_mean = tf.reduce_mean(delta, axis=0)
        delta_std = tf.math.reduce_std(delta, axis=0)
    else:
        delta_mean = tf.zeros_like(mean)
        delta_std = tf.zeros_like(std)

    embedding = tf.concat(
        [mean, std, maxv, delta_mean, delta_std],
        axis=0
    )

    return embedding.numpy()

# =========================
# Public API function
# =========================
def predict_language(audio_path):

    y = load_audio(audio_path)

    if y is None:
        return {"language":"unknown","confidence":0}

    window_size = 16000 * 3
    step = 16000 * 2

    all_probs = []

    for start in range(0, len(y) - window_size + 1, step):

        segment = y[start:start + window_size]

        features = extract_features(segment)
        if features is None:
            continue

        features = features.reshape(1, -1)

        probs = model.predict_proba(features)[0]

        all_probs.append(probs)

    if len(all_probs) == 0:
        return {"language": "unknown", "confidence": 0}

    avg_probs = np.mean(all_probs, axis=0)

    pred_index = np.argmax(avg_probs)

    confidence = float(np.max(avg_probs))

# predicted language
    language = label_encoder.inverse_transform([pred_index])[0]

# sort probabilities descending
    sorted_idx = np.argsort(avg_probs)[::-1]

    alternatives = []

    for idx in sorted_idx:
        lang = label_encoder.inverse_transform([idx])[0]

        if lang != language:
            alternatives.append(lang)

        if len(alternatives) == 2:
            break

    return {
        "language": language,
        "confidence": confidence,
        "alternatives": alternatives,
        "probabilities": avg_probs.tolist(),
        "labels": label_encoder.classes_.tolist()
    }
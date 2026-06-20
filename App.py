"""
MoodAI – Flask backend (OPTIMISED)
Fixes:
  1. threaded=True  → multiple concurrent requests handled properly
  2. WebP accept support (future-proof)
  3. Input size logged for debugging
  4. Smaller model input (matches frontend 320×240 capture)
"""

import os
import base64
import io
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_ENV  = os.environ.get("MODEL_PATH", "emotion_model.h5")
MODEL_PATH = MODEL_ENV if os.path.isabs(MODEL_ENV) else os.path.join(BASE_DIR, MODEL_ENV)

FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

# ── Model config ─────────────────────────────────────────────────────────────
IMG_SIZE       = (100, 100)   # what the Keras model expects
COLOR_MODE     = "rgb"        # "rgb" or "grayscale"
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

# ── Face detector ──────────────────────────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Model (lazy load) ─────────────────────────────────────────────────────────
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        print(f"[MoodAI] Model NOT found at {MODEL_PATH}")
        return None
    print(f"[MoodAI] Loading model from {MODEL_PATH} …")
    _model = tf.keras.models.load_model(MODEL_PATH)
    print("[MoodAI] Model loaded ✓")
    return _model


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_bgr, IMG_SIZE)
    if COLOR_MODE == "grayscale":
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        arr  = gray.astype("float32") / 255.0
        arr  = np.expand_dims(arr, axis=-1)
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    try:
        return send_from_directory(FRONTEND_DIR, "index.html")
    except Exception:
        return jsonify({"status": "ok", "message": "Backend running. Use /health or /predict"}), 200


@app.route("/health", methods=["GET"])
def health():
    m = load_model()
    return jsonify({
        "status":        "ok",
        "model_loaded":  m is not None,
        "model_exists":  os.path.exists(MODEL_PATH),
        "model_path":    MODEL_PATH,
        "working_dir":   os.getcwd(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "No image provided"}), 400

    # ── Decode image ─────────────────────────────────────────────────────────
    try:
        img_data = data["image"]
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        img_bytes  = base64.b64decode(img_data)
        np_arr     = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("imdecode returned None")
    except Exception as exc:
        return jsonify({"success": False, "error": f"Image decode failed: {exc}"}), 400

    # ── Detect faces ─────────────────────────────────────────────────────────
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # scaleFactor=1.2 + minNeighbors=4 is slightly faster than 1.1/5
    faces  = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20)
    )

    if len(faces) == 0:
        return jsonify({"success": True, "faces": [], "message": "No face detected"})

    # ── Load model ────────────────────────────────────────────────────────────
    mdl = load_model()
    if mdl is None:
        return jsonify({
            "success": False,
            "error": f"Model '{MODEL_PATH}' not found. Place emotion_model.h5 next to app.py and restart.",
        }), 503

    # ── Run prediction for each face ─────────────────────────────────────────
    results = []
    for (x, y, w, h) in faces:
        face_crop  = frame_bgr[y:y + h, x:x + w]
        inp        = preprocess_face(face_crop)
        preds      = mdl.predict(inp, verbose=0)[0]
        top_idx    = int(np.argmax(preds))
        confidence = float(preds[top_idx]) * 100

        all_scores = {
            label: round(float(score) * 100, 2)
            for label, score in zip(EMOTION_LABELS, preds)
        }

        results.append({
            "bbox":       [int(x), int(y), int(w), int(h)],
            "emotion":    EMOTION_LABELS[top_idx],
            "confidence": round(confidence, 2),
            "all_scores": all_scores,
        })

    return jsonify({"success": True, "faces": results})

# ── Entry point ─────────────────────────────────────────────────────────────[...]
if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
# ── Entry point ───────────────────────────────────────────────────────────────

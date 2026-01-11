import os
import base64
import json
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Optional TensorFlow import; app still starts if missing
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.efficientnet import preprocess_input
    TF_AVAILABLE = True
    TF_IMPORT_ERROR = None
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e
    tf = None
    load_model = None
    preprocess_input = None

BASE_DIR = os.path.dirname(__file__)
# Load environment variables from .env if present (override empty env vars)
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "model", "best_grape_model.h5"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
PORT = int(os.getenv("PORT", "7860"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "")

# Supported plants and per-plant class names
PLANTS = ["grape", "tomato", "potato"]
CLASS_NAMES_MAP = {
    "grape": [
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy"
    ],
    # Leave these empty for now; you will paste them later
    "tomato": [
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ],
    "potato": [
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
    ],
}

def _model_path_for(plant: str):
    """Return model file path for a given plant."""
    plant = plant.lower()
    return os.path.join(BASE_DIR, "model", f"best_{plant}_model.h5")

app = Flask(__name__)

# Cache models per plant
_models = {}
_labels_map = {}
_model_load_errors = {}


def _to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    """Encode raw image bytes as a data URL suitable for image_url."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_openrouter_ai(plant: str, top_class: str, confidence: float, image_bytes: bytes):
    """Send prediction and image to OpenRouter (Allen AI model) and return advice text.

    Returns: (text, error) where one will be None depending on success.
    """
    if not OPENROUTER_API_KEY:
        return None, "Missing OPENROUTER_API_KEY in environment"

    data_url = _to_data_url(image_bytes, mime="image/png")

    prompt_text = (
        f"Plant: {plant}\n"
        f"Predicted disease: {top_class}\n"
        f"Confidence: {confidence:.4f}\n\n"
        "Please provide: (1) likely further symptoms to look for, (2) possible and natural remedies. "
        "Include an agriculture helpline number serving Tamil Nadu (e.g., Kisan Call Centre 1800-180-1551). "
        "Respond in English only."
    )

    payload = {
        "model": "allenai/molmo-2-8b:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_SITE_TITLE:
        headers["X-Title"] = OPENROUTER_SITE_TITLE

    try:
        resp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=30,
        )
        if resp.status_code != 200:
            return None, f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        # Try to extract text content broadly
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content, None
        # Some providers return a list of parts
        if isinstance(content, list):
            text_parts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            text = "\n".join([t for t in text_parts if t])
            return text or None, None if text else "Empty AI response"
        return None, "Unexpected AI response format"
    except Exception as e:
        return None, f"OpenRouter error: {e}"


def translate_to_tamil(text: str):
    """Translate given English text to Tamil using googletrans. Returns (translated, error)."""
    if not text:
        return None, None
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, dest='ta')
        return result.text, None
    except Exception as e:
        return None, f"Translation error: {e}"


def _load_labels(plant: str):
    """Load optional labels from model/{plant}_labels.txt (one per line)."""
    labels_path = os.path.join(BASE_DIR, "model", f"{plant}_labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
            return labels
    # Fallback to generic labels.txt if present
    generic = os.path.join(BASE_DIR, "model", "labels.txt")
    if os.path.exists(generic):
        with open(generic, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
            return labels
    return None


def ensure_model_loaded(plant: str):
    """Lazy-load the model and labels for a specific plant."""
    plant = (plant or "grape").lower()
    if plant not in PLANTS:
        _model_load_errors[plant] = f"Unsupported plant '{plant}'. Supported: {PLANTS}"
        return

    if _models.get(plant) is not None:
        return

    # TensorFlow not available
    if not TF_AVAILABLE:
        _model_load_errors[plant] = f"TensorFlow import error: {TF_IMPORT_ERROR}"
        return

    model_path = _model_path_for(plant)
    if not os.path.exists(model_path):
        _model_load_errors[plant] = f"Model file not found at {model_path}"
        return

    try:
        # compile=False avoids needing training-only custom objects
        model = load_model(model_path, compile=False)
        _models[plant] = model
        _labels_map[plant] = _load_labels(plant)
        _model_load_errors[plant] = None
    except Exception as e:
        _model_load_errors[plant] = f"Failed to load model: {e}"


@app.route("/health", methods=["GET"])
def health():
    # Report per-plant model availability
    availability = {}
    for plant in PLANTS:
        model_path = _model_path_for(plant)
        availability[plant] = {
            "exists": os.path.exists(model_path),
            "path": model_path,
            "loaded": _models.get(plant) is not None,
            "error": _model_load_errors.get(plant)
        }
    return jsonify({
        "status": "ok",
        "tf_available": TF_AVAILABLE,
        "models": availability,
    })


@app.route("/predict", methods=["POST"]) 
def predict():
    plant = (request.form.get("plant") or request.args.get("plant") or "grape").lower()
    ensure_model_loaded(plant)
    model = _models.get(plant)
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "details": _model_load_errors.get(plant),
            "plant": plant,
        }), 500

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded. Send multipart/form-data with field 'file'."
        }), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file or filename."}), 400

    try:
        from PIL import Image
        # Read original bytes for AI forwarding
        original_bytes = file.read()
        file.stream.seek(0)
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        x = np.array(img, dtype=np.float32)

        # EfficientNetB0 preprocessing
        if preprocess_input is not None:
            x = preprocess_input(x)
        else:
            x = x / 255.0

        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        preds = preds[0].tolist()

        response = {"predictions": preds}
        # Attach top prediction with class name
        top_idx = int(np.argmax(preds))
        top_score = preds[top_idx]
        
        # Use per-plant class names; fall back to labels file
        class_names = CLASS_NAMES_MAP.get(plant) or []
        if not class_names or len(class_names) != len(preds):
            class_names = _labels_map.get(plant)
        
        response["top"] = {
            "index": top_idx,
            "class_name": class_names[top_idx] if class_names else None,
            "score": top_score,
        }
        
        # Include all class names if available
        if class_names:
            response["class_names"] = class_names

        # Call OpenRouter AI for advice
        ai_text, ai_error = call_openrouter_ai(
            plant=plant,
            top_class=response["top"]["class_name"],
            confidence=response["top"]["score"],
            image_bytes=original_bytes,
        )
        advice_ta, ta_error = translate_to_tamil(ai_text) if ai_text else (None, None)
        response["ai"] = {"advice": ai_text, "error": ai_error, "advice_tamil": advice_ta, "translate_error": ta_error}

        return jsonify(response)
    except Exception as e:
        return jsonify({
            "error": "Failed to process image or predict",
            "details": str(e),
        }), 500


@app.route("/ui", methods=["GET"])
def ui():
    """Simple upload UI for manual testing."""
    # You can update the UI to send a 'plant' field via dropdown
    return render_template("ui.html", image_size=IMAGE_SIZE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
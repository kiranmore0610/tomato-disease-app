"""
Flask Web App — Tomato Disease Diagnosis for Farmers
=======================================================
A farmer opens this page on their phone browser, takes/uploads a photo of
a tomato leaf, and instantly gets:
  - The predicted disease (or healthy)
  - A Grad-CAM heatmap showing what the AI focused on
  - A Marathi text explanation
  - A Marathi audio file they can tap to listen to

Install dependencies:
    pip install flask tensorflow opencv-python matplotlib gTTS --break-system-packages

Run locally:
    python app.py
Then open http://localhost:5000 in a browser (or http://<your-PC-IP>:5000
from a phone on the same WiFi network to test before real deployment).
"""

import os
import uuid
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, render_template, jsonify, send_from_directory

from marathi_communication import build_marathi_message, speak_marathi

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
MODEL_PATH = "tomato_disease_model.h5"
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

CLASS_NAMES = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
    "Septoria_leaf_spot", "Spider_mites Two-spotted_spider_mite",
    "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
    "healthy", "powdery_mildew",
]

YIELD_LOSS_TABLE = {
    "Bacterial_spot": (10, 50), "Early_blight": (20, 50), "Late_blight": (30, 70),
    "Leaf_Mold": (10, 50), "Septoria_leaf_spot": (20, 60),
    "Spider_mites Two-spotted_spider_mite": (10, 30), "Target_Spot": (20, 40),
    "Tomato_Yellow_Leaf_Curl_Virus": (30, 90), "Tomato_mosaic_virus": (10, 30),
    "healthy": (0, 0), "powdery_mildew": (10, 30),
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)

# ----------------------------------------------------------------------
# LOAD MODEL ONCE AT STARTUP (not per-request — much faster)
# ----------------------------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)


def find_last_conv_layer(model):
    base = model.get_layer(index=1)
    for layer in reversed(base.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            return base, layer.name
    raise ValueError("Could not find a convolutional layer for Grad-CAM.")


BASE_MODEL, LAST_CONV_LAYER_NAME = find_last_conv_layer(model)
print(f"Model loaded. Using layer for Grad-CAM: {LAST_CONV_LAYER_NAME}")


# ----------------------------------------------------------------------
# GRAD-CAM CORE (same logic as gradcam_explainability.py)
# ----------------------------------------------------------------------
def make_gradcam_heatmap(img_array):
    grad_model = tf.keras.models.Model(
        inputs=BASE_MODEL.input,
        outputs=[BASE_MODEL.get_layer(LAST_CONV_LAYER_NAME).output, BASE_MODEL.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(img_array)
        x = model.get_layer(index=2)(base_output)
        x = model.get_layer(index=4)(x)
        preds = model.get_layer(index=6)(x)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(pred_index), preds.numpy()[0]


def overlay_gradcam(img_path, heatmap, save_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # cv2's built-in JET colormap replaces matplotlib here — same red/blue
    # visual result, but removes matplotlib as a dependency entirely,
    # which cuts meaningful memory (matplotlib pulls in a large font/
    # rendering stack that isn't needed for a single colormap lookup).
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 1 - alpha, jet_heatmap, alpha, 0)
    cv2.imwrite(save_path, superimposed)


# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/debug/memory")
def debug_memory():
    """Diagnostic endpoint: reports current process memory usage in MB.
    Visit this URL directly in your browser (e.g.
    https://your-app.onrender.com/debug/memory) to see whether the app
    is close to Render's free-tier 512MB limit. Remove this route once
    the memory issue is resolved — it's for debugging only."""
    import psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    return jsonify({
        "memory_used_mb": round(mem_mb, 1),
        "render_free_tier_limit_mb": 512,
        "percent_of_limit": round((mem_mb / 512) * 100, 1),
    })


CONFIDENCE_THRESHOLD = 0.60  # below this, treat as "uncertain / not a valid leaf photo"
GREEN_PIXEL_THRESHOLD = 0.15  # at least 15% of the image should be plant-green

NOT_A_LEAF_MESSAGE_MARATHI = (
    "हा फोटो टोमॅटोच्या पानाचा दिसत नाही. कृपया एका पानाचा स्पष्ट, जवळून फोटो "
    "काढा आणि पुन्हा प्रयत्न करा."
)


def looks_like_a_leaf(img_path):
    """Quick color-based check: does this image contain enough green/plant
    coloring to plausibly be a leaf? Runs before the AI model, so it's fast
    and catches obviously wrong photos (faces, walls, random objects)."""
    img = cv2.imread(img_path)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Green/yellow-green hue range covers healthy leaves AND many disease
    # discolorations (yellowing, browning still has some green remaining
    # in a real leaf photo's background/other leaves).
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / mask.size

    return green_ratio >= GREEN_PIXEL_THRESHOLD


@app.route("/diagnose", methods=["POST"])
def diagnose():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    file = request.files["photo"]
    unique_id = uuid.uuid4().hex[:8]
    upload_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.jpg")
    file.save(upload_path)

    # --- Pre-check: does this even look like a plant leaf? ---
    if not looks_like_a_leaf(upload_path):
        return jsonify({
            "predicted_class": None,
            "confidence": 0,
            "not_a_leaf": True,
            "marathi_message": NOT_A_LEAF_MESSAGE_MARATHI,
        })

    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(upload_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict + Grad-CAM
    heatmap, pred_index, all_probs = make_gradcam_heatmap(img_array)
    predicted_class = CLASS_NAMES[pred_index]
    confidence = float(all_probs[pred_index])

    # --- Post-check: is the model actually confident? ---
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            "predicted_class": None,
            "confidence": round(confidence * 100, 2),
            "not_a_leaf": True,
            "marathi_message": NOT_A_LEAF_MESSAGE_MARATHI,
        })

    loss_low, loss_high = YIELD_LOSS_TABLE[predicted_class]

    heatmap_path = os.path.join(RESULT_FOLDER, f"{unique_id}_heatmap.jpg")
    overlay_gradcam(upload_path, heatmap, heatmap_path)

    # Marathi text + audio (audio may be None if TTS times out — see
    # speak_marathi's timeout handling; the diagnosis still returns
    # successfully either way)
    marathi_message = build_marathi_message(predicted_class, confidence, (loss_low, loss_high))
    audio_path = os.path.join(RESULT_FOLDER, f"{unique_id}_audio.mp3")
    audio_result = speak_marathi(marathi_message, save_path=audio_path)

    response = {
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "yield_loss_range": [loss_low, loss_high],
        "marathi_message": marathi_message,
        "heatmap_url": "/" + heatmap_path.replace("\\", "/"),
        "audio_url": ("/" + audio_path.replace("\\", "/")) if audio_result else None,
    }
    return jsonify(response)


if __name__ == "__main__":
    # For local testing only. In production (Render), gunicorn runs the
    # app instead of this block — see Procfile.
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

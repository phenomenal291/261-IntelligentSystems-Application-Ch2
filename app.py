"""
Traffic Sign Recognition Web Application
Lightweight Flask Server with KNN Pre-trained Inference
"""

import os
import io
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import joblib

try:
    from traffic_sign_knn_app.train import extract_features_from_image
except ImportError:
    from train import extract_features_from_image

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_traffic_sign_model.pkl")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

# Load pre-trained model on startup
print(f"Loading pre-trained KNN model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run `python train.py` first.")

model_payload = joblib.load(MODEL_PATH)
knn_model = model_payload["model"]
class_names_map = model_payload["class_names"]
classes_list = list(model_payload["classes_"])
print(f"Model loaded successfully! ({len(classes_list)} traffic sign classes)")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns list of preloaded test sample images for one-click demo."""
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for fname in sorted(os.listdir(SAMPLES_DIR)):
            if fname.endswith(".png"):
                cls_key = fname.replace("_sample.png", "")
                samples.append({
                    "filename": fname,
                    "class_key": cls_key,
                    "name": class_names_map.get(cls_key, cls_key),
                    "url": f"/sample_image/{fname}"
                })
    return jsonify({"samples": samples})

@app.route("/sample_image/<filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()
    
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    try:
        # Read and extract features
        img = Image.open(io.BytesIO(file.read()))
        features = extract_features_from_image(img).reshape(1, -1)

        # KNN Inference
        pred_label = knn_model.predict(features)[0]
        pred_name = class_names_map.get(pred_label, pred_label)
        
        # Probabilities and Nearest Neighbors
        probs = knn_model.predict_proba(features)[0]
        confidence = float(np.max(probs) * 100.0)
        
        # Get distances and indices of K nearest neighbors
        dists, indices = knn_model.kneighbors(features)
        
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return jsonify({
            "success": True,
            "predicted_class": pred_name,
            "class_key": pred_label,
            "confidence": round(confidence, 1),
            "inference_time_ms": round(inference_time_ms, 2),
            "k_neighbors_count": len(indices[0]),
            "min_neighbor_distance": round(float(dists[0][0]), 4)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Traffic Sign KNN Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

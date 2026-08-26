"""
Traffic Sign Recognition Web Application
Flask Server with HOG Feature Visualization & Top-K Nearest Neighbor Exemplar Gallery
"""

import os
import io
import time
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

try:
    from traffic_sign_knn_app.train import extract_features_from_image
except ImportError:
    from train import extract_features_from_image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_traffic_sign_model.pkl")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

print(f"Loading pre-trained KNN model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run `python train.py` first.")

model_payload = joblib.load(MODEL_PATH)
knn_model = model_payload["model"]
class_names_map = model_payload["class_names"]
classes_list = list(model_payload["classes_"])
train_image_paths = model_payload.get("train_image_paths", [])
y_train_labels = model_payload.get("y_train", [])
print(f"Model loaded successfully! ({len(classes_list)} classes, {len(train_image_paths)} exemplar images)")

def render_hog_to_base64(hog_image):
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=90)
    ax.imshow(hog_image, cmap='magma')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def image_to_base64(img_path):
    full_path = os.path.join(BASE_DIR, img_path)
    if os.path.exists(full_path):
        with open(full_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
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
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    try:
        # Load image & compute HOG with visualization
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        features, hog_img_arr, resized_img = extract_features_from_image(img, return_hog_image=True)
        feat_vector = features.reshape(1, -1)

        # KNN Inference
        pred_label = knn_model.predict(feat_vector)[0]
        pred_name = class_names_map.get(pred_label, pred_label)
        
        # Probabilities & Confidence
        probs = knn_model.predict_proba(feat_vector)[0]
        confidence = float(np.max(probs) * 100.0)
        
        # Find K=3 Nearest Neighbors & Distances
        k_neighbors = 3
        dists, indices = knn_model.kneighbors(feat_vector, n_neighbors=k_neighbors)
        
        # Build educational neighbor breakdowns
        neighbors_info = []
        for rank, (idx, d) in enumerate(zip(indices[0], dists[0]), start=1):
            n_label = y_train_labels[idx] if idx < len(y_train_labels) else "unknown"
            n_name = class_names_map.get(n_label, n_label)
            n_img_rel = train_image_paths[idx] if idx < len(train_image_paths) else ""
            n_b64 = image_to_base64(n_img_rel) if n_img_rel else ""
            
            weight = round(1.0 / (float(d) + 1e-5), 2)
            neighbors_info.append({
                "rank": rank,
                "class_name": n_name,
                "distance": round(float(d), 4),
                "weight": weight,
                "image_b64": n_b64
            })

        hog_b64 = render_hog_to_base64(hog_img_arr)
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return jsonify({
            "success": True,
            "predicted_class": pred_name,
            "class_key": pred_label,
            "confidence": round(confidence, 1),
            "inference_time_ms": round(inference_time_ms, 1),
            "feature_dim": len(features),
            "hog_image_b64": hog_b64,
            "neighbors": neighbors_info
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Traffic Sign KNN Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

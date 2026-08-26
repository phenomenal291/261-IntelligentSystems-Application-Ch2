"""
Traffic Sign Recognition Web Application
Optimized High-Performance Flask Server
"""

import os
import io
import time
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import joblib

try:
    from traffic_sign_knn_app.train import extract_features_from_image
except ImportError:
    from train import extract_features_from_image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_traffic_sign_model.pkl")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

print(f"Loading pre-trained model payload from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run `python train.py` first.")

model_payload = joblib.load(MODEL_PATH)
X_train = model_payload["X_train"]
y_train = model_payload["y_train"]
class_names_map = model_payload["class_names"]
classes_list = list(model_payload["classes_"])
train_image_paths = model_payload.get("train_image_paths", [])

# Pre-cache all training exemplar images into base64 in memory for instantaneous zero-IO lookups
print("Pre-caching training exemplar thumbnails in memory...")
image_b64_cache = {}
for rel_p in train_image_paths:
    full_p = os.path.join(BASE_DIR, rel_p)
    if os.path.exists(full_p):
        with open(full_p, "rb") as f:
            image_b64_cache[rel_p] = base64.b64encode(f.read()).decode("utf-8")

print(f"Server ready! ({len(X_train)} training vectors, {len(image_b64_cache)} cached thumbnails)")

def render_hog_to_base64_fast(hog_image_arr):
    """Ultra-fast (sub-millisecond) in-memory HOG map encoding via Pillow"""
    hog_norm = (hog_image_arr / (hog_image_arr.max() + 1e-8) * 255).astype(np.uint8)
    pil_img = Image.fromarray(hog_norm, mode='L').resize((128, 128), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG', optimize=False)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

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

    # Read user-customized KNN parameters
    k_val = int(request.form.get("k", 3))
    k_val = max(1, min(k_val, min(15, len(X_train))))
    
    metric = request.form.get("metric", "euclidean").lower()
    if metric not in ["euclidean", "manhattan", "cosine"]:
        metric = "euclidean"
        
    weights = request.form.get("weights", "distance").lower()

    try:
        # 1. Fast Feature Extraction & HOG Map
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        features, hog_img_arr, _ = extract_features_from_image(img, return_hog_image=True)
        feat_vector = features.reshape(1, -1)

        # 2. Vectorized Distance Computation in Feature Space
        dists = pairwise_distances(feat_vector, X_train, metric=metric)[0]

        # 3. Sort & Select Top-K Nearest Neighbors
        sorted_indices = np.argsort(dists)[:k_val]
        top_k_indices = sorted_indices
        top_k_dists = dists[top_k_indices]
        top_k_labels = y_train[top_k_indices]

        # 4. Voting Mechanism (Uniform vs Distance-Weighted)
        eps = 1e-6
        if weights == "distance":
            vote_weights = 1.0 / (top_k_dists + eps)
        else:
            vote_weights = np.ones(k_val, dtype=float)

        # Aggregate class vote scores
        class_scores = {}
        for lbl, w in zip(top_k_labels, vote_weights):
            class_scores[lbl] = class_scores.get(lbl, 0.0) + w

        total_weight_sum = np.sum(vote_weights)
        winning_class_key = max(class_scores, key=class_scores.get)
        winning_score = class_scores[winning_class_key]
        confidence = float((winning_score / total_weight_sum) * 100.0) if total_weight_sum > 0 else 100.0

        # 5. Build K Nearest Neighbor Exemplar Cards (Instant in-memory cache)
        neighbors_info = []
        for rank, (idx, d, w) in enumerate(zip(top_k_indices, top_k_dists, vote_weights), start=1):
            n_label = y_train[idx]
            n_name = class_names_map.get(n_label, n_label)
            rel_p = train_image_paths[idx] if idx < len(train_image_paths) else ""
            n_b64 = image_b64_cache.get(rel_p, "")

            neighbors_info.append({
                "rank": rank,
                "class_name": n_name,
                "class_key": n_label,
                "distance": round(float(d), 4),
                "weight": round(float(w), 3) if weights == "distance" else "1.0",
                "image_b64": n_b64
            })

        hog_b64 = render_hog_to_base64_fast(hog_img_arr)
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return jsonify({
            "success": True,
            "predicted_class": class_names_map.get(winning_class_key, winning_class_key),
            "class_key": winning_class_key,
            "confidence": round(confidence, 1),
            "inference_time_ms": round(inference_time_ms, 1),
            "k": k_val,
            "metric": metric,
            "weights": weights,
            "hog_image_b64": hog_b64,
            "neighbors": neighbors_info
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Optimized Traffic Sign KNN Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

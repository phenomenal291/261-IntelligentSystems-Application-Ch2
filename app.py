"""
Traffic Sign Recognition Web Application
High-Performance Flask Server with 2D Feature Space Map & Voting Analysis
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
    from traffic_sign_knn_app.train import extract_features_from_image, CLASS_COLORS
except ImportError:
    from train import extract_features_from_image, CLASS_COLORS

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
pca = model_payload["pca"]
X_2d = model_payload["X_2d"]
class_names_map = model_payload["class_names"]
class_colors_map = model_payload.get("class_colors", CLASS_COLORS)
classes_list = list(model_payload["classes_"])
train_image_paths = model_payload.get("train_image_paths", [])

# Pre-cache all training exemplar images into base64 in memory
print("Pre-caching training exemplar thumbnails in memory...")
image_b64_cache = {}
for rel_p in train_image_paths:
    full_p = os.path.join(BASE_DIR, rel_p)
    if os.path.exists(full_p):
        with open(full_p, "rb") as f:
            image_b64_cache[rel_p] = base64.b64encode(f.read()).decode("utf-8")

print(f"Server ready! ({len(X_train)} training vectors, {len(image_b64_cache)} cached thumbnails)")

def render_hog_to_base64_fast(hog_image_arr):
    hog_norm = (hog_image_arr / (hog_image_arr.max() + 1e-8) * 255).astype(np.uint8)
    pil_img = Image.fromarray(hog_norm, mode='L').resize((128, 128), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG', optimize=False)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/feature_map", methods=["GET"])
def get_feature_map():
    """Returns static 2D coordinates for all 350 training points in PCA space."""
    points = []
    for i in range(len(X_2d)):
        lbl = y_train[i]
        points.append({
            "x": round(float(X_2d[i, 0]), 3),
            "y": round(float(X_2d[i, 1]), 3),
            "label": lbl,
            "name": class_names_map.get(lbl, lbl),
            "color": class_colors_map.get(lbl, "#3B82F6"),
            "index": i
        })
    return jsonify({
        "points": points,
        "class_names": class_names_map,
        "class_colors": class_colors_map
    })

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

    # User parameters
    k_val = int(request.form.get("k", 3))
    k_val = max(1, min(k_val, min(60, len(X_train))))
    
    metric = request.form.get("metric", "euclidean").lower()
    if metric not in ["euclidean", "manhattan", "cosine"]:
        metric = "euclidean"
        
    weights = request.form.get("weights", "distance").lower()

    try:
        # 1. Feature Extraction & HOG Map
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        features, hog_img_arr, _ = extract_features_from_image(img, return_hog_image=True)
        feat_vector = features.reshape(1, -1)

        # 2. 2D PCA Projection of Query Point
        q_2d_arr = pca.transform(feat_vector)[0]
        q_2d = [round(float(q_2d_arr[0]), 3), round(float(q_2d_arr[1]), 3)]

        # 3. Distance Computation
        dists = pairwise_distances(feat_vector, X_train, metric=metric)[0]

        # 4. Top-K Sorting
        sorted_indices = np.argsort(dists)[:k_val]
        top_k_indices = sorted_indices
        top_k_dists = dists[top_k_indices]
        top_k_labels = y_train[top_k_indices]

        # 5. Dual Voting Breakdown Comparison (Uniform vs Distance)
        eps = 1e-6
        # A: Uniform votes
        uniform_scores = {}
        for lbl in top_k_labels:
            uniform_scores[lbl] = uniform_scores.get(lbl, 0) + 1
        
        uniform_list = []
        for lbl, cnt in sorted(uniform_scores.items(), key=lambda x: x[1], reverse=True):
            uniform_list.append({
                "class_key": lbl,
                "class_name": class_names_map.get(lbl, lbl),
                "score": cnt,
                "percentage": round((cnt / k_val) * 100.0, 1),
                "color": class_colors_map.get(lbl, "#3B82F6")
            })

        # B: Distance-weighted votes
        dist_weights = 1.0 / (top_k_dists + eps)
        dist_scores = {}
        for lbl, w in zip(top_k_labels, dist_weights):
            dist_scores[lbl] = dist_scores.get(lbl, 0.0) + w
        
        total_w_sum = np.sum(dist_weights)
        dist_list = []
        for lbl, sc in sorted(dist_scores.items(), key=lambda x: x[1], reverse=True):
            dist_list.append({
                "class_key": lbl,
                "class_name": class_names_map.get(lbl, lbl),
                "score": round(float(sc), 2),
                "percentage": round((sc / total_w_sum) * 100.0, 1) if total_w_sum > 0 else 100.0,
                "color": class_colors_map.get(lbl, "#3B82F6")
            })

        # Active Winner
        active_list = dist_list if weights == "distance" else uniform_list
        winning_class_key = active_list[0]["class_key"]
        confidence = active_list[0]["percentage"]

        # 6. Build Top-K Neighbor Exemplars
        neighbors_info = []
        for rank, (idx, d, w) in enumerate(zip(top_k_indices, top_k_dists, dist_weights), start=1):
            n_label = y_train[idx]
            n_name = class_names_map.get(n_label, n_label)
            rel_p = train_image_paths[idx] if idx < len(train_image_paths) else ""
            n_b64 = image_b64_cache.get(rel_p, "")

            neighbors_info.append({
                "rank": rank,
                "index": int(idx),
                "x2d": round(float(X_2d[idx, 0]), 3),
                "y2d": round(float(X_2d[idx, 1]), 3),
                "class_name": n_name,
                "class_key": n_label,
                "color": class_colors_map.get(n_label, "#3B82F6"),
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
            "confidence": confidence,
            "inference_time_ms": round(inference_time_ms, 1),
            "k": k_val,
            "metric": metric,
            "weights": weights,
            "q_2d": q_2d,
            "hog_image_b64": hog_b64,
            "voting_comparison": {
                "uniform": uniform_list,
                "distance": dist_list
            },
            "neighbors": neighbors_info
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Traffic Sign KNN Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

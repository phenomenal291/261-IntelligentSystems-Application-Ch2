"""
Traffic Sign Recognition Web Application
High-Performance Flask Server with 2D Feature Space Map & Voting Analysis
Powered by 52-Class Kaggle Traffic Signs Dataset
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
    from traffic_sign_knn_app.train import extract_features_from_image, CLASS_COLORS, CLASS_NAMES
except ImportError:
    from train import extract_features_from_image, CLASS_COLORS, CLASS_NAMES

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
pca_feat = model_payload.get("pca_feat")
pca_2d = model_payload.get("pca_2d")
X_2d_sub = model_payload.get("X_2d_sub", model_payload.get("X_2d"))
y_2d_sub = model_payload.get("y_2d_sub", y_train)
class_names_map = model_payload.get("class_names", CLASS_NAMES)
class_colors_map = model_payload.get("class_colors", CLASS_COLORS)
classes_list = list(model_payload.get("classes_", np.unique(y_train)))
exemplar_b64 = model_payload.get("exemplar_b64", {})

print(f"Server ready! ({len(X_train)} training vectors, {len(classes_list)} classes)")

def render_hog_to_base64_fast(hog_image_arr):
    if hog_image_arr is None:
        return ""
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
    """Returns static 2D coordinates for canvas display."""
    points = []
    for i in range(len(X_2d_sub)):
        lbl = str(y_2d_sub[i])
        points.append({
            "x": round(float(X_2d_sub[i, 0]), 3),
            "y": round(float(X_2d_sub[i, 1]), 3),
            "label": lbl,
            "name": class_names_map.get(lbl, lbl),
            "color": class_colors_map.get(lbl, "#3B82F6"),
            "index": int(i)
        })
    return jsonify({
        "points": points,
        "class_names": class_names_map,
        "class_colors": class_colors_map
    })

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns curated subset of quick test sample chips."""
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

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)

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
    
    metric = request.form.get("metric", "cosine").lower()
    if metric not in ["euclidean", "manhattan", "cosine"]:
        metric = "cosine"
        
    weights = request.form.get("weights", "distance").lower()

    try:
        # 1. Feature Extraction & HOG Map
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        features_raw, hog_img_arr, _ = extract_features_from_image(img, return_hog_image=True)
        feat_raw_vec = features_raw.reshape(1, -1)

        # 2. PCA Projection (384-dim feature vector)
        if pca_feat is not None:
            feat_vec = pca_feat.transform(feat_raw_vec).astype(np.float32)
        else:
            feat_vec = feat_raw_vec

        # 3. 2D PCA Projection of Query Point for Canvas
        if pca_2d is not None:
            q_2d_arr = pca_2d.transform(feat_vec)[0]
            q_2d = [round(float(q_2d_arr[0]), 3), round(float(q_2d_arr[1]), 3)]
        else:
            q_2d = [0.0, 0.0]

        # 4. Distance Computation
        dists = pairwise_distances(feat_vec, X_train, metric=metric)[0]

        # 5. Top-K Sorting
        sorted_indices = np.argsort(dists)[:k_val]
        top_k_indices = sorted_indices
        top_k_dists = dists[top_k_indices]
        top_k_labels = y_train[top_k_indices]

        # 6. Dual Voting Breakdown Comparison (Uniform vs Distance)
        eps = 1e-6
        # A: Uniform votes
        uniform_scores = {}
        for lbl in top_k_labels:
            lbl_str = str(lbl)
            uniform_scores[lbl_str] = uniform_scores.get(lbl_str, 0) + 1
        
        uniform_list = []
        for lbl_str, cnt in sorted(uniform_scores.items(), key=lambda x: x[1], reverse=True):
            uniform_list.append({
                "class_key": lbl_str,
                "class_name": class_names_map.get(lbl_str, lbl_str),
                "score": int(cnt),
                "percentage": round(float((cnt / k_val) * 100.0), 1),
                "color": class_colors_map.get(lbl_str, "#3B82F6")
            })

        # B: Distance-weighted votes
        dist_weights = 1.0 / (top_k_dists.astype(float) + eps)
        dist_scores = {}
        for lbl, w in zip(top_k_labels, dist_weights):
            lbl_str = str(lbl)
            dist_scores[lbl_str] = dist_scores.get(lbl_str, 0.0) + float(w)
        
        total_w_sum = float(np.sum(dist_weights))
        dist_list = []
        for lbl_str, sc in sorted(dist_scores.items(), key=lambda x: x[1], reverse=True):
            dist_list.append({
                "class_key": lbl_str,
                "class_name": class_names_map.get(lbl_str, lbl_str),
                "score": round(float(sc), 2),
                "percentage": round(float((sc / total_w_sum) * 100.0), 1) if total_w_sum > 0 else 100.0,
                "color": class_colors_map.get(lbl_str, "#3B82F6")
            })

        # Active Winner
        active_list = dist_list if weights == "distance" else uniform_list
        winning_class_key = str(active_list[0]["class_key"])
        confidence = float(active_list[0]["percentage"])

        # 7. Build Top-K Neighbor Exemplars
        neighbors_info = []
        for rank, (idx, d, w) in enumerate(zip(top_k_indices, top_k_dists, dist_weights), start=1):
            n_label = str(y_train[idx])
            n_name = class_names_map.get(n_label, n_label)
            n_b64 = exemplar_b64.get(n_label, "")

            # 2D coordinates for canvas ray tracing
            x2d_val = round(float(model_payload["X_2d"][idx, 0]), 3) if "X_2d" in model_payload else 0.0
            y2d_val = round(float(model_payload["X_2d"][idx, 1]), 3) if "X_2d" in model_payload else 0.0

            neighbors_info.append({
                "rank": int(rank),
                "index": int(idx),
                "x2d": x2d_val,
                "y2d": y2d_val,
                "class_name": n_name,
                "class_key": n_label,
                "color": class_colors_map.get(n_label, "#3B82F6"),
                "distance": round(float(d), 4),
                "weight": round(float(w), 3) if weights == "distance" else "1.0",
                "image_b64": n_b64
            })

        hog_b64 = render_hog_to_base64_fast(hog_img_arr)
        inference_time_ms = float((time.perf_counter() - t0) * 1000.0)

        return jsonify({
            "success": True,
            "predicted_class": class_names_map.get(winning_class_key, winning_class_key),
            "class_key": winning_class_key,
            "confidence": round(confidence, 1),
            "inference_time_ms": round(inference_time_ms, 1),
            "k": int(k_val),
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

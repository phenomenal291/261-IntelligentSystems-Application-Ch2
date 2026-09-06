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
from skimage.feature import hog
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_traffic_sign_model.pkl")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# WSGI Prefix Middleware for Vercel Serverless Routing
class PrefixMiddleware(object):
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # If proxy passed original URI in HTTP headers, restore it if rewritten to index.py
        forwarded_uri = (
            environ.get("HTTP_X_FORWARDED_URI")
            or environ.get("HTTP_X_NOW_ROUTE_MATCHES")
            or environ.get("RAW_URI")
            or environ.get("REQUEST_URI")
        )
        if forwarded_uri and environ.get("PATH_INFO") in ["/api/index.py", "/api/index", "/api"]:
            clean_path = forwarded_uri.split("?")[0]
            if clean_path:
                environ["PATH_INFO"] = clean_path

        path = environ.get("PATH_INFO", "")
        while "//" in path:
            path = path.replace("//", "/")
        environ["PATH_INFO"] = path
        return self.wsgi_app(environ, start_response)

app.wsgi_app = PrefixMiddleware(app.wsgi_app)

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

# 52 Category Color Palette
CLASS_COLORS_DEFAULT = {
    "speed_5": "#EF4444", "speed_15": "#F97316", "speed_30": "#FB923C", "speed_40": "#F59E0B",
    "speed_50": "#EAB308", "speed_60": "#FACC15", "speed_70": "#F59E0B", "speed_80": "#D97706",
    "dont_straight_left": "#DC2626", "priority_road": "#EAB308", "dont_straight": "#DC2626",
    "dont_left": "#DC2626", "dont_left_right": "#DC2626", "dont_right": "#DC2626",
    "dont_overtake_left": "#B91C1C", "no_uturn": "#DC2626", "no_car": "#DC2626",
    "no_horn": "#DC2626", "no_entry": "#EF4444", "no_stopping": "#DC2626",
    "go_straight_right": "#3B82F6", "go_straight": "#3B82F6", "go_left": "#3B82F6",
    "go_left_right": "#3B82F6", "go_right": "#3B82F6", "keep_left": "#2563EB",
    "keep_right": "#2563EB", "roundabout": "#2563EB", "watch_out_cars": "#F59E0B",
    "horn": "#3B82F6", "bicycles_crossing": "#F59E0B", "uturn": "#3B82F6",
    "road_divider": "#F59E0B", "hazard_warning": "#F59E0B", "danger_ahead": "#EF4444",
    "zebra_crossing": "#3B82F6", "cyclists_ahead": "#F59E0B", "children_crossing": "#F59E0B",
    "curve_left": "#F59E0B", "curve_right": "#F59E0B", "road_hazard": "#F59E0B",
    "info_sign": "#3B82F6", "regulatory_sign": "#3B82F6", "go_right_straight": "#3B82F6",
    "go_left_straight": "#3B82F6", "speed_notice": "#3B82F6", "zigzag_curve": "#F59E0B",
    "train_crossing": "#EF4444", "under_construction": "#EA580C", "traffic_notice": "#3B82F6",
    "fences": "#F59E0B", "heavy_accidents": "#DC2626"
}

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
class_names_map = model_payload.get("class_names", {})
class_colors_map = model_payload.get("class_colors", CLASS_COLORS_DEFAULT)
classes_list = list(model_payload.get("classes_", np.unique(y_train)))
exemplar_b64 = model_payload.get("exemplar_b64", {})

# Pre-cache Quick Samples with Base64 Data URIs in memory
PRE_CACHED_SAMPLES = []
if os.path.exists(SAMPLES_DIR):
    for fname in sorted(os.listdir(SAMPLES_DIR)):
        if fname.endswith(".png"):
            cls_key = fname.replace("_sample.png", "")
            fpath = os.path.join(SAMPLES_DIR, fname)
            with open(fpath, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")
            PRE_CACHED_SAMPLES.append({
                "filename": fname,
                "class_key": cls_key,
                "name": class_names_map.get(cls_key, cls_key),
                "url": f"/sample_image/{fname}",
                "data_url": f"data:image/png;base64,{b64_str}"
            })

# Precompute 2D Feature Map Payload
FEATURE_MAP_POINTS = []
if X_2d_sub is not None and len(X_2d_sub) > 0:
    for i in range(len(X_2d_sub)):
        lbl = str(y_2d_sub[i])
        FEATURE_MAP_POINTS.append({
            "x": round(float(X_2d_sub[i, 0]), 2),
            "y": round(float(X_2d_sub[i, 1]), 2),
            "label": lbl,
            "name": class_names_map.get(lbl, lbl),
            "color": class_colors_map.get(lbl, "#3B82F6"),
            "index": int(i)
        })

FEATURE_MAP_PAYLOAD = {
    "points": FEATURE_MAP_POINTS,
    "class_names": class_names_map,
    "class_colors": class_colors_map
}

print(f"Server ready! ({len(X_train)} training vectors, {len(classes_list)} classes, {len(FEATURE_MAP_POINTS)} map points, {len(PRE_CACHED_SAMPLES)} pre-cached samples)")


def extract_features_from_image(img_input, return_hog_image=False):
    """Multi-modal feature extractor matching train.py exactly."""
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = img_input.convert("RGB")

    # Resize to standard 64x64
    img_64 = img.resize((64, 64), Image.Resampling.BILINEAR)
    gray_64 = np.array(img_64.convert("L"), dtype=float) / 255.0

    # 1. Global HOG
    if return_hog_image:
        hog_global, hog_image = hog(
            gray_64,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True
        )
    else:
        hog_global = hog(
            gray_64,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )
        hog_image = None

    # 2. Center Crop HOG (inner 36x36)
    center_crop = gray_64[14:50, 14:50]
    hog_center = hog(
        center_crop,
        orientations=8,
        pixels_per_cell=(6, 6),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )

    # 3. Center Normalized Pixel Template (24x24)
    center_24 = np.array(img.convert("L").resize((24, 24), Image.Resampling.BILINEAR), dtype=float) / 255.0
    c_mean, c_std = center_24.mean(), center_24.std()
    center_norm = (center_24 - c_mean) / (c_std + 1e-5)
    center_vec = center_norm.flatten() * 0.15

    # 4. Color HSV & RGB Statistics
    hsv_64 = np.array(img_64.convert("HSV"), dtype=float) / 255.0
    hue_hist, _ = np.histogram(hsv_64[:, :, 0], bins=8, range=(0.0, 1.0), density=True)
    sat_mean = np.mean(hsv_64[:, :, 1])
    val_mean = np.mean(hsv_64[:, :, 2])

    red_mask = ((hsv_64[:, :, 0] < 0.08) | (hsv_64[:, :, 0] > 0.92)) & (hsv_64[:, :, 1] > 0.25)
    blue_mask = (hsv_64[:, :, 0] >= 0.50) & (hsv_64[:, :, 0] <= 0.72) & (hsv_64[:, :, 1] > 0.25)
    yellow_mask = (hsv_64[:, :, 0] >= 0.10) & (hsv_64[:, :, 0] <= 0.20) & (hsv_64[:, :, 1] > 0.25)

    color_feats = np.hstack([
        hue_hist * 0.05,
        [sat_mean * 0.2, val_mean * 0.2,
         red_mask.mean() * 0.4, blue_mask.mean() * 0.4, yellow_mask.mean() * 0.4]
    ])

    feature_vec = np.hstack([hog_global, hog_center, center_vec, color_feats]).astype(np.float32)
    return feature_vec, hog_image, img_64


def render_hog_to_base64_fast(hog_image_arr):
    if hog_image_arr is None:
        return ""
    hog_norm = (hog_image_arr / (hog_image_arr.max() + 1e-8) * 255).astype(np.uint8)
    pil_img = Image.fromarray(hog_norm, mode='L').resize((128, 128), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG', optimize=False)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route("/", methods=["GET", "POST", "OPTIONS"])
@app.route("/api", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/index", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/index.py", methods=["GET", "POST", "OPTIONS"])
def index():
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200
    if request.method == "POST":
        return predict()
    return render_template("index.html", samples=PRE_CACHED_SAMPLES, feature_map=FEATURE_MAP_PAYLOAD)


@app.route("/api/feature_map", methods=["GET", "OPTIONS"])
@app.route("/feature_map", methods=["GET", "OPTIONS"])
def get_feature_map():
    """Returns static 2D coordinates for canvas display."""
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200
    return jsonify(FEATURE_MAP_PAYLOAD)


@app.route("/api/samples", methods=["GET", "OPTIONS"])
@app.route("/samples", methods=["GET", "OPTIONS"])
def get_samples():
    """Returns curated subset of quick test sample chips with embedded base64 data URIs."""
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200
    return jsonify({"samples": PRE_CACHED_SAMPLES})


@app.route("/static/<path:filename>")
@app.route("/api/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/sample_image/<filename>")
@app.route("/api/sample_image/<filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/predict", methods=["POST", "OPTIONS"])
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    t0 = time.perf_counter()
    img = None
    k_val = 3
    metric = "cosine"
    weights = "distance"

    # 1. Parse JSON Payload or Multipart Form Data
    if request.is_json:
        data_json = request.get_json(silent=True) or {}
        b64_str = data_json.get("image_b64", "")
        if b64_str:
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            try:
                img_bytes = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                return jsonify({"success": False, "error": f"Invalid base64 image: {str(e)}"}), 400

        k_val = int(data_json.get("k", 3))
        metric = str(data_json.get("metric", "cosine")).lower()
        weights = str(data_json.get("weights", "distance")).lower()

    elif "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            try:
                img = Image.open(io.BytesIO(file.read())).convert("RGB")
            except Exception as e:
                return jsonify({"success": False, "error": f"Invalid image file: {str(e)}"}), 400

        k_val = int(request.form.get("k", 3))
        metric = str(request.form.get("metric", "cosine")).lower()
        weights = str(request.form.get("weights", "distance")).lower()

    if img is None:
        return jsonify({"success": False, "error": "No valid image provided."}), 400

    # User parameter validation
    k_val = max(1, min(k_val, min(60, len(X_train))))
    if metric not in ["euclidean", "manhattan", "cosine"]:
        metric = "cosine"

    try:
        # 1. Feature Extraction & HOG Map
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

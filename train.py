"""
Traffic Sign Recognition - Model Training Script
Dataset: Kaggle tuanai/traffic-signs-dataset (52 real-world classes)
Extracts multi-modal features (Global HOG + Center HOG + Center Template + Color Statistics),
applies PCA dimensionality reduction for noise reduction & compact payload,
fits 2D PCA embedding for interactive UI visualization,
and exports the model bundle to models/knn_traffic_sign_model.pkl.
"""

import os
import io
import glob
import base64
import time
import shutil
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import joblib

# Dataset Paths
KAGGLE_DATASET_DIR = "/home/phuc/.cache/kagglehub/datasets/tuanai/traffic-signs-dataset/versions/1"
DATA_DIR = os.path.join(KAGGLE_DATASET_DIR, "DATA")
TEST_DIR = os.path.join(KAGGLE_DATASET_DIR, "TEST")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

# 52 Clean Class Definitions & Color Palette
CLASS_DEFINITIONS = {
    0: {"key": "speed_5", "name": "Speed Limit (5 km/h)", "color": "#EF4444"},
    1: {"key": "speed_15", "name": "Speed Limit (15 km/h)", "color": "#F97316"},
    2: {"key": "speed_30", "name": "Speed Limit (30 km/h)", "color": "#FB923C"},
    3: {"key": "speed_40", "name": "Speed Limit (40 km/h)", "color": "#F59E0B"},
    4: {"key": "speed_50", "name": "Speed Limit (50 km/h)", "color": "#EAB308"},
    5: {"key": "speed_60", "name": "Speed Limit (60 km/h)", "color": "#FACC15"},
    6: {"key": "speed_70", "name": "Speed Limit (70 km/h)", "color": "#F59E0B"},
    7: {"key": "speed_80", "name": "Speed Limit (80 km/h)", "color": "#D97706"},
    8: {"key": "dont_straight_left", "name": "Don't Go Straight or Left", "color": "#DC2626"},
    9: {"key": "priority_road", "name": "Priority Road Sign", "color": "#EAB308"},
    10: {"key": "dont_straight", "name": "Don't Go Straight", "color": "#DC2626"},
    11: {"key": "dont_left", "name": "Don't Go Left", "color": "#DC2626"},
    12: {"key": "dont_left_right", "name": "Don't Go Left or Right", "color": "#DC2626"},
    13: {"key": "dont_right", "name": "Don't Go Right", "color": "#DC2626"},
    14: {"key": "dont_overtake_left", "name": "Don't Overtake from Left", "color": "#B91C1C"},
    15: {"key": "no_uturn", "name": "No U-Turn", "color": "#DC2626"},
    16: {"key": "no_car", "name": "No Motor Vehicles", "color": "#DC2626"},
    17: {"key": "no_horn", "name": "No Horn / Sounding Prohibited", "color": "#DC2626"},
    18: {"key": "no_entry", "name": "No Entry", "color": "#EF4444"},
    19: {"key": "no_stopping", "name": "No Stopping", "color": "#DC2626"},
    20: {"key": "go_straight_right", "name": "Go Straight or Right", "color": "#3B82F6"},
    21: {"key": "go_straight", "name": "Go Straight", "color": "#3B82F6"},
    22: {"key": "go_left", "name": "Go Left", "color": "#3B82F6"},
    23: {"key": "go_left_right", "name": "Go Left or Right", "color": "#3B82F6"},
    24: {"key": "go_right", "name": "Go Right", "color": "#3B82F6"},
    25: {"key": "keep_left", "name": "Keep Left", "color": "#2563EB"},
    26: {"key": "keep_right", "name": "Keep Right", "color": "#2563EB"},
    27: {"key": "roundabout", "name": "Roundabout Mandatory", "color": "#2563EB"},
    28: {"key": "watch_out_cars", "name": "Watch Out for Cars", "color": "#F59E0B"},
    29: {"key": "horn", "name": "Sound Horn Mandatory", "color": "#3B82F6"},
    30: {"key": "bicycles_crossing", "name": "Bicycles Crossing", "color": "#F59E0B"},
    31: {"key": "uturn", "name": "U-Turn Permitted", "color": "#3B82F6"},
    32: {"key": "road_divider", "name": "Road Divider Ahead", "color": "#F59E0B"},
    33: {"key": "hazard_warning", "name": "Hazard Warning Sign", "color": "#F59E0B"},
    34: {"key": "danger_ahead", "name": "Danger Ahead / Warning", "color": "#EF4444"},
    35: {"key": "zebra_crossing", "name": "Pedestrian / Zebra Crossing", "color": "#3B82F6"},
    36: {"key": "cyclists_ahead", "name": "Cyclists Ahead", "color": "#F59E0B"},
    37: {"key": "children_crossing", "name": "Children Crossing", "color": "#F59E0B"},
    38: {"key": "curve_left", "name": "Dangerous Curve Left", "color": "#F59E0B"},
    39: {"key": "curve_right", "name": "Dangerous Curve Right", "color": "#F59E0B"},
    40: {"key": "road_hazard", "name": "Road Hazard Warning", "color": "#F59E0B"},
    41: {"key": "info_sign", "name": "Traffic Information Sign", "color": "#3B82F6"},
    42: {"key": "regulatory_sign", "name": "Regulatory Notice", "color": "#3B82F6"},
    43: {"key": "go_right_straight", "name": "Go Right or Straight", "color": "#3B82F6"},
    44: {"key": "go_left_straight", "name": "Go Left or Straight", "color": "#3B82F6"},
    45: {"key": "speed_notice", "name": "Speed Recommendation", "color": "#3B82F6"},
    46: {"key": "zigzag_curve", "name": "Double / ZigZag Curve", "color": "#F59E0B"},
    47: {"key": "train_crossing", "name": "Railway / Train Crossing", "color": "#EF4444"},
    48: {"key": "under_construction", "name": "Road Work / Under Construction", "color": "#EA580C"},
    49: {"key": "traffic_notice", "name": "Traffic Warning Notice", "color": "#3B82F6"},
    50: {"key": "fences", "name": "Fence / Guardrail Warning", "color": "#F59E0B"},
    51: {"key": "heavy_accidents", "name": "Accident Blackspot Area", "color": "#DC2626"},
}

CLASS_NAMES = {v["key"]: v["name"] for v in CLASS_DEFINITIONS.values()}
CLASS_COLORS = {v["key"]: v["color"] for v in CLASS_DEFINITIONS.values()}
CID_TO_KEY = {cid: v["key"] for cid, v in CLASS_DEFINITIONS.items()}
KEY_TO_CID = {v["key"]: cid for cid, v in CLASS_DEFINITIONS.items()}


def extract_features_from_image(img_input, return_hog_image=False):
    """
    Extracts rich multi-modal features:
    1. Global HOG (64x64) -> Outer boundary shape and overall contour.
    2. Center Crop HOG (inner 36x36) -> Inner glyphs, numerals, and arrows.
    3. Center Normalized Pixel Template (24x24) -> Pixel intensity pattern.
    4. Color HSV & RGB Statistics -> Distinguishes red, blue, and yellow sign categories.
    """
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

    # 2. Center Crop HOG (inner 36x36 box where digits and arrows live)
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


def prepare_curated_samples():
    """Selects 6 prominent test samples for clean UI display."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    # Clean existing samples
    for old_f in glob.glob(os.path.join(SAMPLES_DIR, "*.png")):
        try:
            os.remove(old_f)
        except OSError:
            pass

    curated = [
        ("speed_50_sample.png", os.path.join(TEST_DIR, "4", "004_0019_j.png")),
        ("no_entry_sample.png", os.path.join(TEST_DIR, "18", "055_0005_j.png")),
        ("go_right_sample.png", os.path.join(TEST_DIR, "24", "024_0008.png")),
        ("roundabout_sample.png", os.path.join(TEST_DIR, "27", "027_0007_j.png")),
        ("danger_ahead_sample.png", os.path.join(TEST_DIR, "34", "034_1_0002_1_j.png")),
        ("zebra_crossing_sample.png", os.path.join(TEST_DIR, "35", "035_0005_j.png")),
    ]

    for dest_name, src_path in curated:
        if os.path.exists(src_path):
            img = Image.open(src_path).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
            img.save(os.path.join(SAMPLES_DIR, dest_name), "PNG")
            print(f"Copied curated sample: {dest_name}")


def train_and_export():
    print("=" * 70)
    print("🚀 Training KNN Traffic Sign Classifier on Kaggle 52-Class Dataset")
    print("=" * 70)

    # 1. Prepare Curated Samples
    prepare_curated_samples()

    # 2. Extract Training Features
    print(f"\n📂 Loading training dataset from {DATA_DIR}...")
    t0 = time.time()
    X_train_raw = []
    y_train_keys = []
    exemplar_b64 = {}

    class_dirs = sorted(os.listdir(DATA_DIR), key=lambda x: int(x) if x.isdigit() else 999)
    for c_str in class_dirs:
        cid = int(c_str)
        c_key = CID_TO_KEY.get(cid, f"class_{cid}")
        c_folder = os.path.join(DATA_DIR, c_str)
        img_files = glob.glob(os.path.join(c_folder, "*.*"))

        for i, img_p in enumerate(img_files):
            try:
                feat, _, thumb_img = extract_features_from_image(img_p)
                X_train_raw.append(feat)
                y_train_keys.append(c_key)

                # Cache first 2 thumbnails per class as base64 exemplars
                if c_key not in exemplar_b64:
                    buf = io.BytesIO()
                    thumb_img.resize((48, 48), Image.Resampling.BILINEAR).save(buf, format="PNG", optimize=True)
                    exemplar_b64[c_key] = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                pass

    X_train_raw = np.array(X_train_raw, dtype=np.float32)
    y_train_keys = np.array(y_train_keys)
    print(f"✅ Extracted {len(X_train_raw)} training vectors (Dim: {X_train_raw.shape[1]}) in {time.time()-t0:.2f}s")

    # 3. Extract Test Features for Validation
    print(f"\n📂 Loading validation test dataset from {TEST_DIR}...")
    X_test_raw = []
    y_test_keys = []
    for c_str in sorted(os.listdir(TEST_DIR), key=lambda x: int(x) if x.isdigit() else 999):
        cid = int(c_str)
        c_key = CID_TO_KEY.get(cid, f"class_{cid}")
        c_folder = os.path.join(TEST_DIR, c_str)
        for img_p in glob.glob(os.path.join(c_folder, "*.*")):
            try:
                feat, _, _ = extract_features_from_image(img_p)
                X_test_raw.append(feat)
                y_test_keys.append(c_key)
            except Exception:
                pass

    X_test_raw = np.array(X_test_raw, dtype=np.float32)
    y_test_keys = np.array(y_test_keys)
    print(f"✅ Extracted {len(X_test_raw)} test vectors")

    # 4. Fit PCA (384 components for optimal noise reduction & compact size)
    print("\n🧠 Fitting PCA Dimensionality Reduction (n_components=384)...")
    pca_feat = PCA(n_components=384, random_state=42)
    X_train_pca = pca_feat.fit_transform(X_train_raw).astype(np.float32)
    X_test_pca = pca_feat.transform(X_test_raw).astype(np.float32)

    # 5. Fit 2D PCA for Visualization Canvas
    print("🗺️ Fitting 2D PCA Projection for feature space map...")
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_train_pca).astype(np.float32)

    # 6. Evaluate KNN Across K values
    print("\n📊 Evaluating KNN Classifier Performance:")
    for k in [1, 3, 5, 7, 15, 25]:
        knn_eval = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
        knn_eval.fit(X_train_pca, y_train_keys)
        preds = knn_eval.predict(X_test_pca)
        acc = accuracy_score(y_test_keys, preds)
        print(f"  • K={k:2d} (Cosine, Distance-Weighted): Accuracy = {acc*100:.2f}%")

    # 7. Subsample 2D points for smooth web canvas rendering (e.g. ~1,000 points)
    canvas_indices = np.linspace(0, len(X_2d) - 1, min(1200, len(X_2d)), dtype=int)
    X_2d_sub = X_2d[canvas_indices]
    y_2d_sub = y_train_keys[canvas_indices]

    # 8. Package and Export Payload
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "knn_traffic_sign_model.pkl")

    payload = {
        "X_train": X_train_pca,
        "y_train": y_train_keys,
        "pca_feat": pca_feat,
        "pca_2d": pca_2d,
        "X_2d": X_2d,
        "X_2d_sub": X_2d_sub,
        "y_2d_sub": y_2d_sub,
        "classes_": np.unique(y_train_keys),
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
        "exemplar_b64": exemplar_b64
    }

    print(f"\n💾 Serializing model payload to {out_path} (compress=3)...")
    joblib.dump(payload, out_path, compress=3)
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"✅ Model saved successfully! File size: {file_size_mb:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    train_and_export()

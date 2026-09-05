"""
Traffic Sign Recognition - Model Training Script
Extracts multi-modal features (Global HOG + Center Details + Color Statistics),
fits PCA 2D embedding, and exports payload to models/knn_traffic_sign_model.pkl.
"""

import os
import glob
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

CLASS_NAMES = {
    "stop": "Stop Sign",
    "speed_30": "Speed Limit (30 km/h)",
    "speed_50": "Speed Limit (50 km/h)",
    "speed_80": "Speed Limit (80 km/h)",
    "yield": "Yield / Give Way",
    "no_entry": "No Entry",
    "turn_right": "Turn Right Ahead",
    "turn_left": "Turn Left Ahead",
    "ahead_only": "Ahead Only",
    "pedestrian": "Pedestrian Crossing"
}

CLASS_COLORS = {
    "stop": "#DC2626",        # Red
    "speed_30": "#F97316",     # Orange
    "speed_50": "#F59E0B",     # Amber
    "speed_80": "#EAB308",     # Yellow
    "yield": "#84CC16",        # Lime
    "no_entry": "#EF4444",     # Crimson
    "turn_right": "#3B82F6",   # Blue
    "turn_left": "#06B6D4",    # Cyan
    "ahead_only": "#6366F1",   # Indigo
    "pedestrian": "#8B5CF6"    # Violet
}

def extract_features_from_image(img_input, return_hog_image=False):
    """
    Extracts rich, discriminative multi-modal features:
    1. Global HOG (64x64) -> Boundary shape and overall geometry.
    2. Center Crop HOG (inner 36x36) -> Inner symbol / numbers contours.
    3. Center Normalized Pixel Template (24x24) -> Exact digit/arrow shapes.
    4. Full Image Downsample (16x16) -> Global luminosity distribution.
    5. Color HSV & RGB Statistics -> Distinguishes red, blue, and yellow traffic signs.
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

    # 3. Center Pixel Template (24x24 normalized)
    center_pil = Image.fromarray((center_crop * 255).astype(np.uint8)).resize((24, 24), Image.Resampling.BILINEAR)
    center_arr = np.array(center_pil, dtype=float) / 255.0
    center_norm = (center_arr - np.mean(center_arr)) / (np.std(center_arr) + 1e-6)
    center_flat = center_norm.flatten()

    # 4. Global Low-Res Luminosity (16x16 normalized)
    img_16 = np.array(img_64.convert("L").resize((16, 16)), dtype=float) / 255.0
    img_16_norm = ((img_16 - np.mean(img_16)) / (np.std(img_16) + 1e-6)).flatten()

    # 5. Color Features (HSV Hue histogram + Color channel ratios)
    hsv = np.array(img_64.convert("HSV"), dtype=float)
    h = hsv[:, :, 0] / 255.0
    s = hsv[:, :, 1] / 255.0
    h_hist, _ = np.histogram(h, bins=16, range=(0, 1), weights=s)
    if np.sum(h_hist) > 0:
        h_hist = h_hist / np.sum(h_hist)

    rgb = np.array(img_64, dtype=float) / 255.0
    r_mean = float(np.mean(rgb[:, :, 0]))
    g_mean = float(np.mean(rgb[:, :, 1]))
    b_mean = float(np.mean(rgb[:, :, 2]))
    color_stats = np.array([r_mean, g_mean, b_mean, r_mean - b_mean, b_mean - r_mean])

    # Concatenate features with balanced importance weights
    features = np.concatenate([
        hog_global * 1.0,
        hog_center * 3.0,
        center_flat * 2.5,
        img_16_norm * 1.5,
        h_hist * 4.0,
        color_stats * 3.0
    ])

    if return_hog_image:
        return features, hog_image, img_64
    return features

def load_dataset(data_dir):
    features = []
    labels = []
    image_paths = []
    class_folders = sorted(os.listdir(data_dir))

    for cls_name in class_folders:
        folder_path = os.path.join(data_dir, cls_name)
        if not os.path.isdir(folder_path):
            continue

        img_list = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        for img_p in img_list:
            feat = extract_features_from_image(img_p)
            features.append(feat)
            labels.append(cls_name)
            rel_p = os.path.relpath(img_p, os.path.dirname(os.path.abspath(__file__)))
            image_paths.append(rel_p)

    return np.array(features), np.array(labels), image_paths

def train_and_export():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "train")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Step 1: Extracting multi-modal features from training images...")
    X, y, img_paths = load_dataset(data_dir)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features across {len(np.unique(y))} classes.")

    # Train / Test split evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    for k in [1, 3, 5]:
        knn_eval = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance')
        knn_eval.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, knn_eval.predict(X_test))
        print(f"Hold-out Test Accuracy (K={k}, Euclidean): {test_acc * 100:.2f}%")

    # Fit 2D PCA for visual feature space map
    print("Step 2: Fitting PCA 2D projection for interactive feature space map...")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    # Cast to float32 for compact memory and fast network transfer (<12 MB)
    X_f32 = X.astype(np.float32)
    X_2d_f32 = X_2d.astype(np.float32)

    model_payload = {
        "X_train": X_f32,
        "y_train": y,
        "pca": pca,
        "X_2d": X_2d_f32,
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
        "classes_": np.unique(y),
        "feature_dim": X.shape[1],
        "train_image_paths": img_paths
    }

    model_path = os.path.join(models_dir, "knn_traffic_sign_model.pkl")
    joblib.dump(model_payload, model_path, compress=3)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Step 3: Compact Model payload ({file_size_mb:.2f} MB), PCA 2D embedding, and metadata exported to: {model_path}")

if __name__ == "__main__":
    train_and_export()

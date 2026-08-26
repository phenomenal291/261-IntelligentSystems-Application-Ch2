"""
Traffic Sign Recognition - Model Training Script
Extracts HOG (Histogram of Oriented Gradients) features and trains a KNN Classifier.
Exports the trained model and exemplar metadata to models/knn_traffic_sign_model.pkl.
"""

import os
import glob
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
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

def extract_features_from_image(img_input, return_hog_image=False):
    """
    Extracts HOG (Histogram of Oriented Gradients) feature vector from image.
    Optionally returns the 2D HOG visualization array.
    """
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = img_input.convert("RGB")

    # Resize to standard 32x32
    img_resized = img.resize((32, 32), Image.Resampling.BILINEAR)
    gray_arr = np.array(img_resized.convert("L"), dtype=float) / 255.0

    if return_hog_image:
        features, hog_image = hog(
            gray_arr,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True
        )
        return features, hog_image, img_resized
    else:
        features = hog(
            gray_arr,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )
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
            # Store relative path for portability
            rel_p = os.path.relpath(img_p, os.path.dirname(os.path.abspath(__file__)))
            image_paths.append(rel_p)

    return np.array(features), np.array(labels), image_paths

def train_and_export():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "train")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Step 1: Loading images & extracting HOG descriptors...")
    X, y, img_paths = load_dataset(data_dir)
    print(f"Loaded {len(X)} samples with {X.shape[1]} HOG features across {len(np.unique(y))} classes.")

    # Train / Test split
    indices = np.arange(len(X))
    idx_tr, idx_te = train_test_split(indices, test_size=0.15, random_state=42, stratify=y)

    X_train, y_train = X[idx_tr], y[idx_tr]
    X_test, y_test = X[idx_te], y[idx_te]
    train_paths = [img_paths[i] for i in idx_tr]

    print("\nStep 2: Fitting KNeighborsClassifier (K=3, Euclidean, Distance-Weighted)...")
    knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean', weights='distance')
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Hold-out Test Accuracy: {test_acc * 100:.2f}%")

    # Fit final model on all data so all exemplars are available for nearest neighbor retrieval
    knn_full = KNeighborsClassifier(n_neighbors=3, metric='euclidean', weights='distance')
    knn_full.fit(X, y)

    model_payload = {
        "model": knn_full,
        "class_names": CLASS_NAMES,
        "classes_": knn_full.classes_,
        "feature_dim": X.shape[1],
        "train_image_paths": img_paths,
        "y_train": y
    }

    model_path = os.path.join(models_dir, "knn_traffic_sign_model.pkl")
    joblib.dump(model_payload, model_path)
    print(f"\nStep 3: Model & exemplar gallery successfully exported to: {model_path}")

if __name__ == "__main__":
    train_and_export()

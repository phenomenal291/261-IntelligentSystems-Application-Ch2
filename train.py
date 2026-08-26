"""
Traffic Sign Recognition - Model Training Script
Extracts HOG (Histogram of Oriented Gradients) features and trains a KNN Classifier.
Exports the trained model to models/knn_traffic_sign_model.pkl.
"""

import os
import glob
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Human-readable labels dictionary
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

def extract_features_from_image(img_input):
    """
    Extracts HOG (Histogram of Oriented Gradients) feature vector from image.
    Supports PIL Image or file path.
    """
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = img_input.convert("RGB")

    # Resize to fixed standard dimensions (32 x 32)
    img_resized = img.resize((32, 32), Image.Resampling.BILINEAR)
    
    # Convert to grayscale numpy array
    gray_arr = np.array(img_resized.convert("L"), dtype=float) / 255.0

    # Extract HOG features (orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
    hog_features = hog(
        gray_arr,
        orientations=8,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    return hog_features

def load_dataset(data_dir):
    features = []
    labels = []
    class_folders = sorted(os.listdir(data_dir))
    
    for cls_name in class_folders:
        folder_path = os.path.join(data_dir, cls_name)
        if not os.path.isdir(folder_path):
            continue
        
        image_paths = glob.glob(os.path.join(folder_path, "*.png"))
        for img_p in image_paths:
            feat = extract_features_from_image(img_p)
            features.append(feat)
            labels.append(cls_name)

    return np.array(features), np.array(labels)

def train_and_export():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "train")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Step 1: Loading images & extracting HOG features...")
    X, y = load_dataset(data_dir)
    print(f"Loaded {len(X)} samples with {X.shape[1]} HOG features each across {len(np.unique(y))} classes.")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nStep 2: Training K-Nearest Neighbors Classifier (K=3, Euclidean, Distance-Weighted)...")
    knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean', weights='distance')
    knn_model.fit(X_train, y_train)

    # Evaluation
    y_pred = knn_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    cv_scores = cross_val_score(knn_model, X, y, cv=5)
    print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 100:.2f}%)")

    # Package model metadata
    model_payload = {
        "model": knn_model,
        "class_names": CLASS_NAMES,
        "classes_": knn_model.classes_,
        "feature_dim": X.shape[1]
    }

    model_path = os.path.join(models_dir, "knn_traffic_sign_model.pkl")
    joblib.dump(model_payload, model_path)
    print(f"\nStep 3: Model successfully exported to: {model_path}")

if __name__ == "__main__":
    train_and_export()

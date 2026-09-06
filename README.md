# Traffic Sign Recognition System using K-Nearest Neighbors (KNN)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)

An end-to-end, lightweight web application that classifies road traffic signs in real time using **Multi-Modal Feature Extraction (Global & Center HOG + Template Matching + Color Statistics + PCA)** and an interactive **K-Nearest Neighbors (KNN)** classifier.

---

## 📌 Project Overview

This project demonstrates a real-world, practical computer vision application of the K-Nearest Neighbors algorithm:
1. **Dataset**: Trained on **5,683 real-world traffic sign images** across **52 distinct classes** from the Kaggle Traffic Signs Dataset (`tuanai/traffic-signs-dataset`).
2. **Multi-Modal Feature Engineering**:
   - **Global HOG ($64\times 64$)**: Extracts outer boundary geometry and geometric contour.
   - **Center Crop HOG ($36\times 36$)**: Captures inner glyphs, digits (speed limits), and arrows.
   - **Normalized Center Template ($24\times 24$)**: Resolves fine numeric differences.
   - **HSV & RGB Statistics**: Hue histograms, saturation, and color masks to distinguish red, blue, and yellow traffic signs.
   - **PCA Dimensionality Reduction ($n=384$)**: Filters noise and compresses payload to **10.5 MB** for instant serverless startup.
3. **Dynamic KNN Hyperparameter Controls**:
   - **$K$ (Number of Neighbors)**: Slider range $K \in [1, 45]$ with instant presets ($K=1, 3, 7, 15, 25, 45$).
   - **Distance Metric**: Euclidean ($L_2$), Manhattan ($L_1$), and Cosine Distance.
   - **Voting Scheme**: Distance-Weighted ($w_i = 1/d_i$) and Uniform Majority voting.
4. **Interactive Visualizations**:
   - **2D PCA Feature Space Canvas**: Interactive 2D projection with query rays and neighbor radius bounding circle.
   - **HOG Feature Map Visualizer**: Real-time gradient orientation map rendering.
   - **Dual Voting Comparison**: Side-by-side breakdown of Uniform vs. Distance-weighted confidence.
   - **Nearest Neighbors Gallery**: Exemplar cards with thumbnail images, distances, and vote weights.

---

## 🚦 Model Performance

- **Training Samples**: 5,683 images
- **Test Validation Set**: 433 images
- **Accuracy**:
  - $K=1$: **94.00%**
  - $K=3$: **93.53%**
  - $K=5$: **92.84%**
  - $K=7$: **91.92%**
- **Inference Latency**: **~15 - 35 ms** per image

---

## 🗂️ Repository Structure

```
traffic_sign_knn_app/
├── data/
│   └── test_samples/          # Curated quick test sample images for 1-click evaluation
├── models/
│   └── knn_traffic_sign_model.pkl  # Compressed pre-trained KNN & PCA model (10.5 MB)
├── static/
│   ├── css/
│   │   └── style.css          # Modern dark-mode responsive UI styling
│   └── js/
│       └── script.js          # Interactive canvas, clipboard paste, & AJAX inference client
├── templates/
│   └── index.html             # Semantic drag-and-drop & paste interface
├── train.py                   # Model training and feature extraction pipeline
├── app.py                     # High-performance Flask inference server
├── requirements.txt           # Python package dependencies
├── vercel.json                # Vercel serverless deployment configuration
└── README.md                  # Project documentation & reproduction guide
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Clone the Repository
```bash
git clone https://github.com/phenomenal291/261-IntelligentSystems-Application-Ch2.git
cd 261-IntelligentSystems-Application-Ch2
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-train the Model
The pre-trained model is already included in `models/knn_traffic_sign_model.pkl`. To re-train:
```bash
python train.py
```

### 5. Launch the Web Application
```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`.

---

## 🎓 Academic Context

Developed for **CO3061: Intelligent Systems (Hệ thống Thông minh)** at **Ho Chi Minh City University of Technology (HCMUT)**.

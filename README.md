# Traffic Sign Recognition System using K-Nearest Neighbors (KNN)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)

An end-to-end, lightweight web application that classifies road traffic signs in real time using **Multi-Modal Feature Extraction (Global & Center HOG + Template Matching + Color Statistics + PCA)** and an interactive **K-Nearest Neighbors (KNN)** classifier.

---

## 📌 Project Overview

1. **Dataset**: Trained on **5,683 real-world images** across **52 distinct classes** from Kaggle (`tuanai/traffic-signs-dataset`).
2. **Feature Engineering**:
   - **Global HOG ($64\times 64$)**: Boundary geometry and contour.
   - **Center Crop HOG ($36\times 36$)**: Inner glyphs, numerals, and arrows.
   - **Normalized Center Template ($24\times 24$)**: Pixel intensity pattern.
   - **HSV / RGB Statistics**: Hue histograms and color ratios (red, blue, yellow).
   - **PCA ($n=384$)**: Filters noise and compresses payload to **10.5 MB**.
3. **Interactive Controls & Visualizations**:
   - **Hyperparameters**: $K \in [1, 45]$, Distance Metric (Cosine, Euclidean, Manhattan), and Voting (Distance-Weighted vs. Uniform).
   - **2D Feature Map Canvas**: Real-time projection with query rays and neighbor radius circle.
   - **HOG Feature Map**: Live gradient orientation visualizer.
   - **Voting Breakdown**: Side-by-side comparison of Uniform vs. Distance-weighted confidence.
   - **Exemplars Gallery**: Visual top-$K$ nearest neighbors with distances and weights.

---

## 📊 Evaluation & Benchmarks

Evaluated on **433 hold-out real-world test images** across **52 classes**:

### 1. Overall Performance ($K=3$, Cosine, Distance-Weighted)
| Metric | Value |
| :--- | :---: |
| **Accuracy** | **93.53%** (94.00% at $K=1$) |
| **Precision (Weighted)** | **95.19%** |
| **Recall (Weighted)** | **93.53%** |
| **F1-Score (Weighted)** | **93.43%** |

### 2. Hyperparameter Comparison ($K$, Metric & Voting)
| Metric | $K$ | Voting | Accuracy | Precision | F1-Score |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Cosine** | **1** | **Distance / Uniform** | **94.00%** | **95.57%** | **93.88%** |
| **Cosine** | **3** | **Distance-Weighted** | **93.53%** | **95.19%** | **93.43%** |
| Cosine | 3 | Uniform Majority | 84.53% | 88.43% | 84.15% |
| Cosine | 7 | Distance-Weighted | 91.92% | 94.10% | 91.71% |
| Cosine | 15 | Distance-Weighted | 89.15% | 92.60% | 88.79% |
| Cosine | 25 | Distance-Weighted | 89.61% | 92.59% | 89.26% |
| **Euclidean** | 1 | Distance / Uniform | 93.53% | 95.62% | 93.53% |
| Euclidean | 3 | Distance-Weighted | 92.15% | 94.28% | 91.95% |
| Euclidean | 7 | Distance-Weighted | 90.53% | 93.39% | 90.47% |
| **Manhattan** | 1 | Distance / Uniform | 92.38% | 94.33% | 92.25% |
| Manhattan | 3 | Distance-Weighted | 90.53% | 92.86% | 90.29% |

*Insight: Distance-weighted voting ($w = 1/d$) prevents distant majority classes from dominating, maintaining high accuracy even at large $K$.*

### 3. Inference Latency (Single CPU Query)
| Pipeline Stage | Mean Latency | 95th Percentile |
| :--- | :---: | :---: |
| 1. Feature Extraction (HOG + Color) | 10.05 ms | 11.41 ms |
| 2. PCA Projection (384 dims) | 1.38 ms | 5.34 ms |
| 3. KNN Search & Sorting (5.6k points) | 5.95 ms | 8.36 ms |
| **Total End-to-End Latency** | **17.38 ms** | **22.40 ms** (~57.5 FPS) |

---

## 🗂️ Repository Structure

```
traffic_sign_knn_app/
├── data/
│   └── test_samples/          # Curated quick test sample images
├── models/
│   └── knn_traffic_sign_model.pkl  # Compressed model payload (10.5 MB)
├── static/
│   ├── css/
│   │   └── style.css          # Dark-mode responsive styling
│   └── js/
│       └── script.js          # Interactive canvas & client logic
├── templates/
│   └── index.html             # UI interface layout
├── train.py                   # Model training & feature pipeline
├── app.py                     # High-performance Flask server
├── requirements.txt           # Python dependencies
├── vercel.json                # Vercel serverless deployment config
└── README.md                  # Project documentation & evaluation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Clone the Repository
```bash
git clone https://github.com/phenomenal291/261-IntelligentSystems-Application-Ch2.git
cd 261-IntelligentSystems-Application-Ch2
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Model
```bash
python train.py
```

### 4. Launch the Web Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## 🎓 Academic Context

Developed for **CO3061: Intelligent Systems (Hệ thống Thông minh)** at **Ho Chi Minh City University of Technology (HCMUT)**.

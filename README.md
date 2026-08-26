# Traffic Sign Recognition System using K-Nearest Neighbors (KNN)
### Practical Computer Vision Application • CO3061 Intelligent Systems

An end-to-end, lightweight web application that classifies road traffic signs in real time using **Histogram of Oriented Gradients (HOG)** feature extraction and a **K-Nearest Neighbors (KNN)** classifier.

---

## 📌 Project Overview

This project demonstrates a real-world, practical use case of the K-Nearest Neighbors algorithm in intelligent transportation systems and autonomous driving:
1. **Feature Extraction**: Raw traffic sign images are converted to grayscale and transformed into compact, gradient-orientation feature vectors using **HOG (Histogram of Oriented Gradients)**.
2. **KNN Classification**: A distance-weighted $K=3$ Nearest Neighbors model identifies the closest matching sign category in Euclidean feature space.
3. **Web Interface**: A modern, minimal Flask interface allows users to drag-and-drop custom images or click preloaded samples to get instant predictions with confidence scores and latency metrics.

---

## 🗂️ Repository Structure

```
traffic_sign_knn_app/
├── data/
│   ├── train/                 # Labeled training images organized by class
│   └── test_samples/          # Quick test samples for 1-click evaluation
├── models/
│   └── knn_traffic_sign_model.pkl  # Pre-trained KNN model payload
├── static/
│   ├── css/
│   │   └── style.css          # Minimalist responsive UI styling
│   └── js/
│       └── script.js          # Client-side asynchronous upload & prediction
├── templates/
│   └── index.html             # Clean drag-and-drop upload interface
├── train.py                   # Standalone data processing & model training script
├── app.py                     # Lightweight Flask inference server
├── generate_dataset.py        # Synthetic traffic sign dataset generator
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation & reproduction guide
```

---

## 🚦 Supported Traffic Sign Classes

The system currently recognizes 10 international traffic sign categories:
1. **Stop Sign** (`stop`)
2. **Speed Limit 30 km/h** (`speed_30`)
3. **Speed Limit 50 km/h** (`speed_50`)
4. **Speed Limit 80 km/h** (`speed_80`)
5. **Yield / Give Way** (`yield`)
6. **No Entry** (`no_entry`)
7. **Turn Right Ahead** (`turn_right`)
8. **Turn Left Ahead** (`turn_left`)
9. **Ahead Only** (`ahead_only`)
10. **Pedestrian Crossing** (`pedestrian`)

---

## ⚙️ Prerequisites

- **Python Version**: Python `3.9`, `3.10`, `3.11`, `3.12`, or `3.13`
- **OS**: Linux, macOS, or Windows

---

## 🚀 Quickstart & Reproduction Guide

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd traffic_sign_knn_app
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-train the Model
The pre-trained model is already included in `models/knn_traffic_sign_model.pkl`. To train from scratch:
```bash
python train.py
```
*Expected Output: ~97% Test Accuracy and ~94% 5-Fold Cross-Validation Accuracy.*

### 5. Launch the Web Application
```bash
python app.py
```

### 6. Access the Localhost Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🌐 Web Interface Features

- **Drag & Drop Upload Zone**: Drag any PNG/JPG traffic sign image directly onto the drop area or click to browse.
- **1-Click Test Samples**: Instantly test pre-loaded signs (Stop, Speed Limits, Yield, Arrows) with a single click.
- **Real-Time Inference**: Asynchronous `POST /predict` API calls return predictions without page reloads.
- **Detailed Metrics**: Displays the detected sign name, prediction confidence percentage, minimum nearest neighbor distance, and inference latency in milliseconds (~5-15 ms).

---

## 📡 REST API Documentation

### `POST /predict`
Upload an image to get traffic sign classification.

- **Request**: `multipart/form-data` with key `image` (binary file)
- **Response**:
```json
{
  "success": true,
  "predicted_class": "Stop Sign",
  "class_key": "stop",
  "confidence": 100.0,
  "inference_time_ms": 7.42,
  "k_neighbors_count": 3,
  "min_neighbor_distance": 0.1824
}
```

---

## 🎓 Academic Context

Developed for **CO3061: Intelligent Systems (Hệ thống Thông minh)** at **Ho Chi Minh City University of Technology (HCMUT)** to demonstrate practical instance-based learning (K-Nearest Neighbors) combined with classical computer vision feature descriptors.

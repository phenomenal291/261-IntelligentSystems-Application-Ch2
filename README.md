# Traffic Sign Recognition System using K-Nearest Neighbors (KNN)

---

## 📌 Project Overview

This project demonstrates a real-world, practical use case of the K-Nearest Neighbors algorithm in intelligent transportation systems and autonomous driving:
1. **Feature Extraction**: Raw traffic sign images are converted to grayscale and transformed into compact, gradient-orientation feature vectors using **HOG (Histogram of Oriented Gradients)** (1,568 dimensions).
2. **Dynamic KNN Classification**: Full interactive control over:
   - **$K$ (Number of Neighbors)**: Adjust $K \in [1, 9]$ on the fly.
   - **Distance Metric**: Switch between Euclidean ($L_2$), Manhattan ($L_1$), and Cosine Distance.
   - **Voting Scheme**: Switch between Distance-Weighted ($w_i = 1/d_i$) and Uniform Majority voting.
3. **Educational Insights**:
   - **HOG Feature Visualizer**: Inspect the raw input sign vs. its computed HOG gradient orientation map.
   - **Top-$K$ Nearest Training Exemplars Gallery**: Visually displays the actual $K$ nearest training images retrieved from the training set, with their Euclidean distances and vote weights.
4. **Intuitive User Interface**: Drag-and-drop, 1-click test samples, and **direct clipboard paste (`Ctrl+V`)** support.

---

## 🗂️ Repository Structure

```
traffic_sign_knn_app/
├── data/
│   ├── train/                 # Labeled training images organized by class (350 samples)
│   └── test_samples/          # Quick test samples for 1-click evaluation
├── models/
│   └── knn_traffic_sign_model.pkl  # Trained KNN feature vectors & exemplar metadata
├── static/
│   ├── css/
│   │   └── style.css          # Minimalist responsive UI styling
│   └── js/
│       └── script.js          # Asynchronous upload, paste (Ctrl+V), & dynamic KNN client
├── templates/
│   └── index.html             # Semantic HTML5 drag-and-drop & paste interface
├── train.py                   # Standalone feature extraction & model training script
├── app.py                     # Lightweight Flask inference server with dynamic parameters
├── generate_dataset.py        # Synthetic traffic sign dataset generator
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation & reproduction guide
```

---

## 🚦 Supported Traffic Sign Classes

The system recognizes 10 standard traffic sign categories:
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
git clone https://github.com/phenomenal291/261-IntelligentSystems-Application-Ch2
cd 261-IntelligentSystems-Application-Ch2
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
The pre-trained model payload is included in `models/knn_traffic_sign_model.pkl`. To re-train:
```bash
python train.py
```

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

- **Clipboard Paste (`Ctrl+V` / `Cmd+V`)**: Copy any traffic sign image from the web or screenshot tool and paste it directly onto the page.
- **Drag & Drop Upload**: Drag any PNG/JPG traffic sign image directly onto the drop zone or browse local files.
- **1-Click Test Samples**: Instantly test preloaded signs (Stop, Speed Limits, Yield, Arrows, etc.).
- **Live Hyperparameter Tuning**: Adjust $K$, distance metric (Euclidean, Manhattan, Cosine), and voting scheme with immediate dynamic re-evaluation.
- **Top-$K$ Exemplars Gallery**: Visually displays the $K$ closest training images retrieved from the training set, their distance $d(Q, P_i)$, and vote weights.

---

## 📡 REST API Documentation

### `POST /predict`
- **Request Form Data**:
  - `image`: Binary image file (PNG/JPG)
  - `k`: Number of neighbors (Integer, e.g. `3`, `5`, `7`)
  - `metric`: Distance metric (`"euclidean"`, `"manhattan"`, `"cosine"`)
  - `weights`: Voting scheme (`"distance"`, `"uniform"`)
- **Response**:
```json
{
  "success": true,
  "predicted_class": "Stop Sign",
  "class_key": "stop",
  "confidence": 100.0,
  "inference_time_ms": 6.8,
  "k": 3,
  "metric": "euclidean",
  "weights": "distance",
  "hog_image_b64": "...",
  "neighbors": [
    {
      "rank": 1,
      "class_name": "Stop Sign",
      "class_key": "stop",
      "distance": 2.248,
      "weight": 0.445,
      "image_b64": "..."
    }
  ]
}
```

---

## 🎓 Academic Context

Developed for **CO3061: Intelligent Systems** at **Ho Chi Minh City University of Technology (HCMUT)** to demonstrate practical instance-based learning (K-Nearest Neighbors) combined with classical computer vision feature descriptors.

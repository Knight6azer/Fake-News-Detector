# TruthLens — Advanced AI Fake News Detector 🔍

> **Portfolio Project** · Python · Flask · TensorFlow · scikit-learn · NLTK · NLP

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-blue?logo=scikit-learn)](https://scikit-learn.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue)](LICENSE)

A production-ready, full-stack fake news detection web application. Originally built in 3rd semester of engineering and now rebuilt as an advanced portfolio project — featuring dual AI models, a REST API, explainable word highlights, and a stunning dark glassmorphism web interface.

---

## 🌟 Features

| Feature | Description |
|---|---|
| **Dual AI Models** | TF-IDF + Logistic Regression (fast) & BiLSTM + TensorFlow (deep) |
| **Ensemble Mode** | Blends both models — 40% TF-IDF + 60% BiLSTM |
| **Word Highlights** | Visual heatmap showing which words triggered the verdict |
| **Confidence Gauge** | Animated SVG arc showing prediction confidence % |
| **Batch Analysis** | Analyze up to 50 headlines/articles at once |
| **Live Dashboard** | Real-time Chart.js statistics and prediction history |
| **REST API** | 6 clean JSON endpoints for external integration |
| **Auto-Training** | Model trains automatically on first run if no saved model found |
| **Dark/Light Mode** | Glassmorphism UI with theme toggle |
| **Particle Animation** | Dynamic particle network in hero background |

---

## 🏗️ Project Structure

```
Fake News Detection (py)/
├── app.py                    # Flask app + REST API (6 endpoints)
├── requirements.txt          # Python dependencies
├── README.md
├── LICENSE
│
├── model/
│   ├── __init__.py
│   ├── preprocess.py         # NLTK text cleaning, lemmatization, feature extraction
│   ├── train_model.py        # TF-IDF + Logistic Regression trainer
│   ├── deep_model.py         # BiLSTM + Conv1D TensorFlow model trainer
│   ├── predict.py            # Unified prediction pipeline (TF-IDF / LSTM / Ensemble)
│   └── saved/                # Saved model .pkl / .keras files (auto-created on train)
│
├── static/
│   ├── css/style.css         # Dark glassmorphism theme, animations, responsive
│   └── js/main.js            # Prediction logic, particle system, Chart.js, history
│
├── templates/
│   └── index.html            # Single-page app (Hero, Analyzer, Dashboard, Batch)
│
└── data/                     # Dataset folder (auto-created, gitignored)
    └── news.csv              # 6,335 REAL/FAKE news samples (auto-downloaded)
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Knight6azer/Fake-News-Detector.git
cd Fake-News-Detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```

> ✅ The app **automatically downloads the dataset and trains the model** on first run. No manual setup needed!

### 4. Open in Browser
```
http://localhost:5000
```

---

## 🤖 Models

### ⚡ TF-IDF + Logistic Regression *(Default — Fast)*
| Property | Value |
|---|---|
| **Achieved Accuracy** | **94.16%** on 6,335 real samples |
| **Speed** | ~23ms per prediction |
| **Features** | 50,000 n-grams (1–3-gram), sublinear TF, balanced class weights |
| **Dataset** | [news.csv — GeeksforGeeks](https://media.geeksforgeeks.org/wp-content/uploads/20250319152540940977/news.csv) |
| **Best for** | Fast, explainable predictions with word-level importance |

```
              precision  recall  f1-score  support
  REAL          0.95      0.93    0.94       634
  FAKE          0.93      0.95    0.94       633
  accuracy                        0.94      1267
```

### 🧠 BiLSTM + Conv1D *(Deep Learning)*
| Property | Value |
|---|---|
| **Architecture** | Embedding → SpatialDropout → Conv1D → MaxPool → BiLSTM → Dense |
| **Embedding dim** | 128 |
| **Max sequence length** | 300 tokens |
| **Best for** | Context-aware, long-range dependency detection |

Train with:
```bash
python -m model.deep_model
```

### 🔮 Ensemble *(Best Accuracy)*
- Weighted average: **40% TF-IDF + 60% BiLSTM**
- Select in the UI after training both models

---

## 🌐 REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Health check & model availability |
| `POST` | `/api/predict` | Single article prediction |
| `POST` | `/api/batch` | Batch prediction (up to 50 texts) |
| `GET` | `/api/stats` | Session-level statistics |
| `GET` | `/api/history` | Recent predictions (last 100) |
| `POST` | `/api/train` | Trigger model retraining |

### Example — Single Prediction

**Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "SHOCKING: 5G towers cause COVID-19, government hides truth",
    "text": "Leaked documents prove bombshell deep state conspiracy. Share before they delete this!",
    "model": "tfidf"
  }'
```

**Response:**
```json
{
  "id": "d556dd4f",
  "label": "FAKE",
  "confidence": 96.84,
  "model_used": "TF-IDF + Logistic Regression",
  "elapsed_ms": 23.4,
  "word_highlights": [
    {"word": "SHOCKING", "score": 1.0027, "direction": "fake"},
    {"word": "Share",    "score": 4.3078, "direction": "fake"},
    {"word": "Leaked",   "score": 1.081,  "direction": "fake"}
  ]
}
```

---

## 🧪 Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python 3.13, Flask 3.1, Flask-CORS |
| **ML / NLP** | scikit-learn 1.7, TensorFlow 2.20, NLTK 3.9, NumPy, pandas, joblib |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JS, Chart.js 4.4 |
| **Fonts** | Google Fonts — Space Grotesk + Inter |
| **Design** | Dark theme, CSS animations, SVG gauge, particle system |

---

## 💡 How It Works

```
Input Text
    │
    ▼
[Preprocessing]
  Lowercase → Remove URLs/HTML → Remove stopwords → Lemmatize (NLTK)
    │
    ▼
[Feature Extraction]
  TF-IDF (50K n-gram features) — title weighted 3×
    │
    ▼
[Classification]
  Logistic Regression  ──┐
  BiLSTM + Conv1D     ──┤──→ Ensemble (40/60 weighted)
    │
    ▼
[Result]
  Label (FAKE/REAL) + Confidence % + Word Importance Highlights
```

---

## 📁 Dataset

- **Source**: [GeeksforGeeks — Fake News Detection](https://www.geeksforgeeks.org/fake-news-detection-model-using-tensorflow-in-python/)
- **Size**: 6,335 articles (3,171 REAL / 3,164 FAKE) — perfectly balanced
- **Auto-downloaded**: Yes, on first `python app.py` run
- **Format**: CSV with `title`, `text`, `label` columns

---

*Originally a 3rd-semester engineering project — rebuilt as a full-stack advanced portfolio showcase.*

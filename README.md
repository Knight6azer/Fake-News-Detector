# TruthLens — Advanced AI Fake News Detector 🔍

> **Portfolio Project** · Python · Flask · TensorFlow · scikit-learn · NLP

A production-ready, full-stack fake news detection web application. This is an advanced upgrade of a 3rd-semester engineering project, now featuring dual AI models, a REST API, and a stunning modern web interface.

---

## 🌟 Features

| Feature | Description |
|---|---|
| **Dual AI Models** | TF-IDF + Logistic Regression (fast) & BiLSTM + TensorFlow (deep) |
| **Ensemble Mode** | Blends both models for maximum accuracy (40/60 weighted) |
| **Word Highlights** | Visual heatmap showing which words triggered the verdict |
| **Confidence Gauge** | Animated SVG arc showing prediction confidence % |
| **Batch Analysis** | Analyze up to 50 headlines at once |
| **Live Dashboard** | Real-time prediction statistics and history |
| **REST API** | Clean JSON endpoints for integration |
| **Dark/Light Mode** | Glassmorphism UI with theme toggle |
| **Particle Animation** | Dynamic particle network in hero background |

---

## 🏗️ Project Structure

```
Fake News Detection (py)/
├── app.py                    # Flask app + REST API
├── requirements.txt
├── README.md
│
├── model/
│   ├── preprocess.py         # Text cleaning, lemmatization, NLTK pipeline
│   ├── train_model.py        # TF-IDF + Logistic Regression trainer
│   ├── deep_model.py         # BiLSTM TensorFlow model trainer
│   ├── predict.py            # Unified prediction pipeline
│   └── saved/                # Saved model files (auto-created)
│
├── static/
│   ├── css/style.css         # Dark glassmorphism theme
│   └── js/main.js            # Frontend logic
│
└── templates/
    └── index.html            # Single-page app
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
python app.py
```
> ✅ The app **automatically trains the model** on first run if no saved model is found!

### 3. Open in Browser
```
http://localhost:5000
```

---

## 🤖 Models

### ⚡ TF-IDF + Logistic Regression
- **Accuracy**: ~95%+
- **Speed**: <50ms per prediction
- **Features**: 50,000 n-grams (1–3), sublinear TF, balanced class weights
- **Best for**: Quick, explainable predictions

### 🧠 BiLSTM + Conv1D (TensorFlow)
- **Accuracy**: ~97%+
- **Architecture**: Embedding → SpatialDropout → Conv1D → MaxPool → BiLSTM → Dense
- **Best for**: Context-aware, sequence-level understanding

### 🔮 Ensemble
- **Method**: Weighted average (40% TF-IDF + 60% BiLSTM)
- **Best for**: Maximum robustness across all news styles

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Health check & model availability |
| `POST` | `/api/predict` | Single article prediction |
| `POST` | `/api/batch` | Batch prediction (up to 50) |
| `GET` | `/api/stats` | Session statistics |
| `GET` | `/api/history` | Recent predictions |
| `POST` | `/api/train` | Trigger model retraining |

### Example: Single Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "SHOCKING: Scientists prove 5G causes COVID",
    "text": "Bombshell leaked documents prove what deep state has hidden...",
    "model": "tfidf"
  }'
```

**Response:**
```json
{
  "id": "a1b2c3d4",
  "label": "FAKE",
  "confidence": 94.7,
  "model_used": "TF-IDF + Logistic Regression",
  "word_highlights": [...],
  "elapsed_ms": 23.4
}
```

---

## 🛠️ Training Your Own Model

```bash
# Train TF-IDF model (auto-downloads dataset)
python model/train_model.py

# Train BiLSTM model (requires TensorFlow)
python model/deep_model.py
```

Dataset: [news.csv from GeeksforGeeks](https://media.geeksforgeeks.org/wp-content/uploads/20250319152540940977/news.csv)

---

## 🧪 Tech Stack

- **Backend**: Python 3.10+, Flask 3.0, Flask-CORS
- **ML/NLP**: scikit-learn, TensorFlow 2.16, NLTK, NumPy, pandas, joblib
- **Frontend**: Vanilla HTML5/CSS3/JS, Chart.js, Google Fonts
- **Design**: Dark Glassmorphism, CSS animations, SVG gauges

---

## 📸 Features Demo

- **Hero Section**: Particle network animation, gradient text, floating cards
- **Analyzer**: Real-time word heatmap, animated confidence gauge
- **Dashboard**: Live Chart.js doughnut chart, prediction history
- **Batch Mode**: Process 50 articles simultaneously

---

*Built as an advanced portfolio upgrade of a 3rd-semester engineering project.*

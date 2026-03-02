"""
Unified prediction interface for Fake News Detection.
Supports TF-IDF (fast) and BiLSTM (deep) models.
Auto-loads whichever models are available.
"""

import os
import sys
import joblib
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
MODEL_PATH      = os.path.join(MODEL_DIR, "tfidf_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
TOKENIZER_PATH  = os.path.join(MODEL_DIR, "lstm_tokenizer.pkl")

# Module-level cache
_tfidf_model      = None
_tfidf_vectorizer = None
_lstm_model       = None
_lstm_tokenizer   = None
_models_loaded    = False


def load_models(force: bool = False):
    """Load all available saved models into memory (cached)."""
    global _tfidf_model, _tfidf_vectorizer, _lstm_model, _lstm_tokenizer, _models_loaded

    if _models_loaded and not force:
        return

    # Load TF-IDF model
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            _tfidf_model      = joblib.load(MODEL_PATH)
            _tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
            print("[✓] TF-IDF model loaded.")
        except Exception as e:
            print(f"[!] TF-IDF load error: {e}")
    
    # Load LSTM model (optional, heavier)
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        try:
            import tensorflow as tf
            _lstm_model     = tf.keras.models.load_model(LSTM_MODEL_PATH)
            _lstm_tokenizer = joblib.load(TOKENIZER_PATH)
            print("[✓] BiLSTM model loaded.")
        except Exception as e:
            print(f"[!] LSTM load error: {e}")

    _models_loaded = True


def is_trained() -> dict:
    """Return status of available models."""
    return {
        'tfidf': os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH),
        'lstm':  os.path.exists(LSTM_MODEL_PATH) and os.path.exists(TOKENIZER_PATH),
    }


def predict(
    text: str,
    title: str = "",
    model_type: str = "tfidf"
) -> dict:
    """
    Run fake news prediction on input text.

    Args:
        text:       The article body / news content.
        title:      Optional article title (used for combined features).
        model_type: 'tfidf' | 'lstm' | 'ensemble'

    Returns:
        dict with keys: label, confidence, word_highlights, model_used, error
    """
    load_models()

    from model.preprocess import combine_title_text, get_word_importance_scores, clean_text

    combined = combine_title_text(title, text) if title else clean_text(text)

    # ── TF-IDF prediction ────────────────────────────────────────────────
    tfidf_result = None
    if _tfidf_model is not None and _tfidf_vectorizer is not None:
        try:
            vec = _tfidf_vectorizer.transform([combined])
            prob = _tfidf_model.predict_proba(vec)[0]  # [P(REAL), P(FAKE)]
            fake_prob = float(prob[1])
            label = 'FAKE' if fake_prob >= 0.5 else 'REAL'
            confidence = fake_prob if label == 'FAKE' else (1.0 - fake_prob)
            tfidf_result = {
                'label': label,
                'confidence': round(confidence * 100, 2),
                'fake_prob': round(fake_prob, 4),
                'model_used': 'TF-IDF + Logistic Regression',
            }
        except Exception as e:
            tfidf_result = {'error': str(e)}

    # ── LSTM prediction ──────────────────────────────────────────────────
    lstm_result = None
    if _lstm_model is not None and _lstm_tokenizer is not None:
        try:
            from model.deep_model import predict_lstm
            lstm_result = predict_lstm(combined, _lstm_model, _lstm_tokenizer)
            lstm_result['model_used'] = 'BiLSTM + Conv1D'
        except Exception as e:
            lstm_result = {'error': str(e)}

    # ── Select result based on model_type ────────────────────────────────
    if model_type == 'lstm' and lstm_result and 'error' not in lstm_result:
        result = lstm_result
    elif model_type == 'ensemble' and tfidf_result and lstm_result \
            and 'error' not in tfidf_result and 'error' not in lstm_result:
        # Weighted ensemble: 40% TF-IDF + 60% LSTM
        ensemble_prob = 0.4 * tfidf_result['fake_prob'] + 0.6 * lstm_result['raw_prob']
        label = 'FAKE' if ensemble_prob >= 0.5 else 'REAL'
        confidence = ensemble_prob if label == 'FAKE' else (1.0 - ensemble_prob)
        result = {
            'label': label,
            'confidence': round(confidence * 100, 2),
            'fake_prob': round(ensemble_prob, 4),
            'model_used': 'Ensemble (TF-IDF + BiLSTM)',
        }
    else:
        # Default to TF-IDF
        if tfidf_result and 'error' not in tfidf_result:
            result = tfidf_result
        elif lstm_result and 'error' not in lstm_result:
            result = lstm_result
        else:
            return {
                'label': 'UNKNOWN',
                'confidence': 0,
                'error': 'No model available. Please run model/train_model.py first.',
                'word_highlights': [],
                'model_used': 'None'
            }

    # ── Word importance highlights ───────────────────────────────────────
    word_highlights = []
    if _tfidf_model is not None and _tfidf_vectorizer is not None:
        try:
            full_text = f"{title} {text}".strip()
            word_highlights = get_word_importance_scores(full_text, _tfidf_vectorizer, _tfidf_model)
        except Exception:
            pass

    result['word_highlights'] = word_highlights
    result['input_length'] = len(text.split())

    return result


if __name__ == "__main__":
    sample_texts = [
        ("Real News Test", "The Federal Reserve held interest rates steady at its latest meeting, citing stable inflation and labor market data."),
        ("Fake News Test", "SHOCKING: Scientists PROVE that 5G towers are causing COVID-19 and the government is HIDING this bombshell truth from you!"),
    ]
    for title, text in sample_texts:
        result = predict(text, title=title)
        print(f"\n[{result.get('label', '?')}] {title}")
        print(f"  Confidence : {result.get('confidence', 0):.1f}%")
        print(f"  Model      : {result.get('model_used', 'N/A')}")

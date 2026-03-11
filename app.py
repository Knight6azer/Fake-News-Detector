"""
Flask REST API backend for Fake News Detection.
Serves the web UI and handles all prediction requests.
"""

import os
import sys
import io
# Fix stdout encoding on Windows so emoji/Unicode prints don't crash
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import time
import uuid
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fakenews-detector-2024-secret-key')
CORS(app)

# ── Global state ──────────────────────────────────────────────────────────────
prediction_history = []   # In-memory history (last 500)
stats = {
    'total_predictions': 0,
    'fake_count': 0,
    'real_count': 0,
    'avg_confidence': 0.0,
    'model_usage': {'tfidf': 0, 'lstm': 0, 'ensemble': 0},
    'started_at': datetime.now(timezone.utc).isoformat(),
}

# ── Lazy model loader ────────────────────────────────────────────────────────
_model_ready = False

def ensure_model():
    """Ensure models are loaded, train if needed."""
    global _model_ready
    if _model_ready:
        return True
    
    from model.predict import is_trained, load_models
    status = is_trained()
    
    if not status['tfidf']:
        print("[!] No trained model found. Training TF-IDF model now...")
        try:
            from model.train_model import train
            train()
        except Exception as e:
            print(f"[✗] Auto-training failed: {e}")
            return False
    
    load_models()
    _model_ready = True
    return True


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def api_status():
    """Health-check and model availability."""
    from model.predict import is_trained
    model_status = is_trained()
    return jsonify({
        'status': 'ok',
        'models': model_status,
        'ready': model_status.get('tfidf', False),
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Predict single article.
    Body: { text, title (optional), model (optional: tfidf|lstm|ensemble) }
    """
    if not ensure_model():
        return jsonify({'error': 'Model not available. Please run train_model.py.'}), 503

    data = request.get_json(silent=True) or {}
    text       = data.get('text', '').strip()
    title      = data.get('title', '').strip()
    model_type = data.get('model', 'tfidf').lower()

    if not text and not title:
        return jsonify({'error': 'Please provide text or title to analyze.'}), 400
    if len(text) + len(title) < 10:
        return jsonify({'error': 'Input too short. Please provide more text.'}), 400

    input_text = text or title
    
    start = time.time()
    try:
        from model.predict import predict
        result = predict(input_text, title=title, model_type=model_type)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    elapsed = round((time.time() - start) * 1000, 1)  # ms

    # Build response
    prediction_id = str(uuid.uuid4())[:8]
    response = {
        'id': prediction_id,
        'label': result.get('label', 'UNKNOWN'),
        'confidence': result.get('confidence', 0),
        'model_used': result.get('model_used', 'TF-IDF'),
        'word_highlights': result.get('word_highlights', [])[:50],  # cap at 50
        'input_length': result.get('input_length', len(input_text.split())),
        'elapsed_ms': elapsed,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    # Update stats
    label = response['label']
    stats['total_predictions'] += 1
    if label == 'FAKE':
        stats['fake_count'] += 1
    elif label == 'REAL':
        stats['real_count'] += 1
    
    # Rolling average confidence
    n = stats['total_predictions']
    stats['avg_confidence'] = round(
        (stats['avg_confidence'] * (n - 1) + response['confidence']) / n, 2
    )
    stats['model_usage'][model_type] = stats['model_usage'].get(model_type, 0) + 1

    # History (keep last 100)
    history_entry = {
        'id': prediction_id,
        'title': (title or input_text)[:80],
        'label': label,
        'confidence': response['confidence'],
        'model': model_type,
        'timestamp': response['timestamp'],
    }
    prediction_history.insert(0, history_entry)
    if len(prediction_history) > 100:
        prediction_history.pop()

    return jsonify(response)


@app.route('/api/batch', methods=['POST'])
def api_batch():
    """
    Batch prediction for multiple texts.
    Body: { texts: [str, ...], model (optional) }
    """
    if not ensure_model():
        return jsonify({'error': 'Model not available.'}), 503

    data = request.get_json(silent=True) or {}
    texts      = data.get('texts', [])
    model_type = data.get('model', 'tfidf').lower()

    if not texts or not isinstance(texts, list):
        return jsonify({'error': 'Provide a JSON array of texts under key "texts".'}), 400
    if len(texts) > 50:
        return jsonify({'error': 'Maximum 50 texts per batch request.'}), 400

    results = []
    from model.predict import predict
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            results.append({'index': i, 'error': 'Empty or invalid text'})
            continue
        try:
            res = predict(text.strip(), model_type=model_type)
            results.append({
                'index': i,
                'text_preview': text[:100],
                'label': res.get('label', 'UNKNOWN'),
                'confidence': res.get('confidence', 0),
                'model_used': res.get('model_used', ''),
            })
            stats['total_predictions'] += 1
            if res.get('label') == 'FAKE':
                stats['fake_count'] += 1
            else:
                stats['real_count'] += 1
        except Exception as e:
            results.append({'index': i, 'error': str(e)})

    summary = {
        'total': len(texts),
        'fake': sum(1 for r in results if r.get('label') == 'FAKE'),
        'real': sum(1 for r in results if r.get('label') == 'REAL'),
        'errors': sum(1 for r in results if 'error' in r),
    }

    return jsonify({'results': results, 'summary': summary})


@app.route('/api/history', methods=['GET'])
def api_history():
    """Return recent prediction history."""
    limit = min(int(request.args.get('limit', 20)), 100)
    return jsonify({
        'history': prediction_history[:limit],
        'total': len(prediction_history)
    })


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Return cumulative prediction statistics."""
    return jsonify(stats)


@app.route('/api/train', methods=['POST'])
def api_train():
    """Trigger model (re)training (admin endpoint)."""
    global _model_ready
    try:
        from model.train_model import train
        _, _, acc = train()
        _model_ready = False
        ensure_model()
        return jsonify({'success': True, 'accuracy': round(acc * 100, 2)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  [+] Fake News Detector - Advanced Portfolio Edition")
    print("=" * 60)
    print("  Starting server at http://localhost:5000")
    print("  Auto-training model if not already trained...")
    print("=" * 60)
    ensure_model()
    app.run(debug=True, host='0.0.0.0', port=5000)

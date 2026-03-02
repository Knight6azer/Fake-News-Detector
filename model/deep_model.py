"""
BiLSTM deep learning model for Fake News Detection using TensorFlow/Keras.

Architecture: Embedding → SpatialDropout → Conv1D → MaxPool → BiLSTM → Dense → Sigmoid

Run to train:
    python model/deep_model.py
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "lstm_tokenizer.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
MAX_VOCAB = 30000
MAX_LENGTH = 300
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10


def build_bilstm_model(vocab_size: int, embedding_dim: int, max_length: int):
    """
    Build BiLSTM architecture:
    Embedding → SpatialDropout1D → Conv1D → MaxPool → BiLSTM → Dropout → Dense → Sigmoid
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, SpatialDropout1D, Conv1D, MaxPooling1D,
        Bidirectional, LSTM, Dropout, Dense, GlobalMaxPooling1D
    )

    model = Sequential([
        Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        SpatialDropout1D(0.3),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        GlobalMaxPooling1D(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    return model


def train_lstm(csv_path: str = None):
    """Train the BiLSTM model and save it."""
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split

    from model.preprocess import combine_title_text
    from model.train_model import download_dataset, load_and_prepare_data

    if csv_path is None:
        csv_path = download_dataset()

    df = load_and_prepare_data(csv_path)

    X = df['combined'].values
    y = df['encoded_label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[→] Tokenizing text for LSTM...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LENGTH, padding='post', truncating='post')
    X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LENGTH, padding='post', truncating='post')

    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    vocab_size = min(len(tokenizer.word_index), MAX_VOCAB)

    print(f"[→] Building BiLSTM model (vocab={vocab_size}, max_len={MAX_LENGTH})...")
    model = build_bilstm_model(vocab_size, EMBEDDING_DIM, MAX_LENGTH)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    print("[→] Training BiLSTM model...")
    history = model.fit(
        X_train_seq, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_seq, y_test),
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
    print(f"\n[✓] BiLSTM Test Accuracy: {acc*100:.2f}%")

    model.save(LSTM_MODEL_PATH)
    joblib.dump(tokenizer, TOKENIZER_PATH)

    print(f"[✓] LSTM model saved: {LSTM_MODEL_PATH}")
    print(f"[✓] Tokenizer saved:  {TOKENIZER_PATH}")

    return model, tokenizer, acc


def load_lstm_model():
    """Load saved BiLSTM model and tokenizer."""
    import tensorflow as tf
    if not os.path.exists(LSTM_MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None
    try:
        model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        tokenizer = joblib.load(TOKENIZER_PATH)
        return model, tokenizer
    except Exception as e:
        print(f"[!] Could not load LSTM model: {e}")
        return None, None


def predict_lstm(text: str, model, tokenizer) -> dict:
    """Run prediction using BiLSTM model."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = 'FAKE' if prob >= 0.5 else 'REAL'
    confidence = prob if label == 'FAKE' else (1.0 - prob)
    return {'label': label, 'confidence': round(confidence * 100, 2), 'raw_prob': round(prob, 4)}


if __name__ == "__main__":
    train_lstm()

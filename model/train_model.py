"""
TF-IDF + Logistic Regression model trainer.
Fast, interpretable, ~95%+ accuracy baseline for Fake News Detection.

Run this script to train and save the model:
    python model/train_model.py
"""

import os
import sys
import requests
import io
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Add parent dir to path so we can import preprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.preprocess import combine_title_text, clean_text

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_URL = "https://media.geeksforgeeks.org/wp-content/uploads/20250319152540940977/news.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "label_classes.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def download_dataset():
    """Download the news dataset from GeeksforGeeks mirror."""
    csv_path = os.path.join(DATA_DIR, "news.csv")
    
    if os.path.exists(csv_path):
        print(f"[OK] Dataset found at {csv_path}")
        return csv_path
    
    print(f"[>>] Downloading dataset from {DATASET_URL} ...")
    try:
        resp = requests.get(DATASET_URL, timeout=60)
        resp.raise_for_status()
        with open(csv_path, 'wb') as f:
            f.write(resp.content)
        print(f"[OK] Dataset downloaded: {csv_path}")
        return csv_path
    except Exception as e:
        print(f"[!!] Download failed: {e}")
        print("[i] Generating synthetic dataset for demonstration...")
        return generate_synthetic_dataset(csv_path)


def generate_synthetic_dataset(csv_path: str) -> str:
    """
    Generate a synthetic balanced dataset for demo/offline use.
    Realistic fake vs real news samples.
    """
    real_samples = [
        ("Scientists confirm climate change accelerating", "A new study published in Nature confirms that global temperatures are rising at an unprecedented rate, citing data from 1,000 weather stations."),
        ("Stock market rises for third consecutive week", "The S&P 500 closed up 0.3% on Friday, marking its third consecutive week of gains, driven by strong corporate earnings."),
        ("Government announces infrastructure spending plan", "The administration unveiled a $1.2 trillion infrastructure package focused on roads, bridges, and broadband internet access."),
        ("New vaccine shows 92% efficacy in trials", "Clinical trials conducted across 40,000 participants show the new mRNA vaccine to be 92% effective against severe disease."),
        ("Supreme Court rules on data privacy case", "The court issued a 6-3 ruling expanding consumer data privacy protections, limiting corporate data collection practices."),
        ("Scientists discover new species of deep sea fish", "Marine biologists from NOAA announced the discovery of a bioluminescent fish species in the Mariana Trench."),
        ("Federal Reserve holds interest rates steady", "The Fed voted 9-0 to maintain the benchmark interest rate, citing stable inflation and employment figures."),
        ("Olympic committee announces 2032 venue plans", "The International Olympic Committee confirmed Brisbane as the 2032 Summer Games host city after extensive bidding process."),
        ("Electric vehicle sales reach record high", "EV sales jumped 40% year-over-year, with battery technology improvements driving longer range at lower cost."),
        ("WHO updates diabetes management guidelines", "New guidelines recommend earlier lifestyle intervention and updated medication thresholds based on large-scale population studies."),
    ] * 50

    fake_samples = [
        ("SHOCKING: Bill Gates admits chips in vaccines, will control minds", "A leaked video PROVES that Bill Gates has admitted to placing microchips in COVID vaccines to track and control the global population. Mainstream media is HIDING this bombshell."),
        ("BREAKING: Trump secretly won 2024 by 50 million votes, expose reveals", "A stunning new report from a patriot whistleblower reveals that 50 MILLION votes were stolen. The deep state is covering this up right now. Share before they ban this!"),
        ("MIRACLE cure: Eating lemon with baking soda destroys cancer in 48 hours", "Doctors HATE this! Scientists have discovered that a simple mixture of lemon and baking soda can obliterate all cancer cells within 48 hours, but Big Pharma is suppressing it."),
        ("EXCLUSIVE: Government spraying chemtrails to make population infertile", "Whistleblower pilot reveals planes are secretly spraying chemical agents to reduce the global birth rate. This is confirmed by classified documents BANNED from public view."),
        ("ALIEN spacecraft lands at Pentagon, military in panic", "Multiple eyewitnesses confirm a UFO landed outside the Pentagon yesterday evening. The government is desperately censoring this explosive story. Share NOW before deletion!"),
        ("5G towers proven to cause COVID-19, scientists say", "A new bombshell study reveals what they don't want you to know: 5G radiation activates dormant viruses in the human body. All mainstream media is hiding this truth."),
        ("EXPOSED: Obama born in Kenya, new certificate discovered", "Unearthed documents reveal that Barack Hussein Obama was absolutely born in Kenya. This shocking expose confirms what patriots have said for years. Media SILENT."),
        ("Scientist discovers human body can run on sunlight, no food needed", "A rogue genius scientist has PROVEN that humans can photosynthesize just like plants using a secret technique governments want banned. THEY don't want you to know this."),
        ("BOMBSHELL: Hollywood stars are lizard people, proof inside", "Exclusive video footage analyzed by facial recognition technology proves beyond any doubt that multiple A-list celebrities are actually shape-shifting reptilians. Watch before removal!"),
        ("Elite globalists plan to collapse economy by Christmas", "Whistleblowers inside the World Economic Forum confirm a conspiracy to deliberately crash the global economy, making citizens dependent on a one-world digital currency they control."),
    ] * 50

    rows = []
    for i, (title, text) in enumerate(real_samples):
        rows.append({'title': title, 'text': text, 'label': 'REAL'})
    for i, (title, text) in enumerate(fake_samples):
        rows.append({'title': title, 'text': text, 'label': 'FAKE'})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Synthetic dataset saved with {len(df)} samples: {csv_path}")
    return csv_path


def load_and_prepare_data(csv_path: str):
    """Load dataset and prepare combined features."""
    print("[>>] Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Handle potential 'Unnamed: 0' column from the GeeksforGeeks dataset
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    
    # Validate columns
    required_cols = ['label']
    if 'title' not in df.columns and 'text' not in df.columns:
        raise ValueError("Dataset must have at least 'title' or 'text' column and 'label' column.")
    
    if 'title' not in df.columns:
        df['title'] = ''
    if 'text' not in df.columns:
        df['text'] = ''
    
    df = df.dropna(subset=['label'])
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    
    print(f"[OK] Loaded {len(df)} samples")
    print(f"[OK] Label distribution:\n{df['label'].value_counts()}")
    
    # Combine title + text with title weighted 3x
    print("[>>] Preprocessing text (this may take a minute)...")
    df['combined'] = df.apply(
        lambda row: combine_title_text(row['title'], row['text']), axis=1
    )
    
    # Encode labels: FAKE=1, REAL=0
    label_map = {'FAKE': 1, 'REAL': 0, 'fake': 1, 'real': 0, '1': 1, '0': 0, 1: 1, 0: 0}
    df['encoded_label'] = df['label'].map(label_map)
    df = df.dropna(subset=['encoded_label'])
    df['encoded_label'] = df['encoded_label'].astype(int)
    
    return df


def train(csv_path: str = None):
    """Full training pipeline."""
    if csv_path is None:
        csv_path = download_dataset()
    
    df = load_and_prepare_data(csv_path)
    
    X = df['combined'].values
    y = df['encoded_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[>>] Training set: {len(X_train)} | Test set: {len(X_test)}")
    
    # TF-IDF vectorizer
    print("[>>] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),       # unigrams, bigrams, trigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,         # log normalization
        strip_accents='unicode',
        analyzer='word',
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Logistic Regression
    print("[>>] Training Logistic Regression classifier...")
    model = LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"  Model Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    
    # Save model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(['REAL', 'FAKE'], LABELS_PATH)
    
    print(f"\n[OK] Model saved to:      {MODEL_PATH}")
    print(f"[OK] Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"[OK] Accuracy: {acc*100:.2f}%")
    
    return model, vectorizer, acc


if __name__ == "__main__":
    train()

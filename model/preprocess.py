"""
Text preprocessing pipeline for Fake News Detection.
Handles cleaning, normalization, and feature engineering.
"""

import re
import string
import nltk
import numpy as np

def download_nltk_data():
    resources = ['stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Words that strongly indicate fake/biased news
SENSATIONAL_WORDS = {
    'shocking', 'breaking', 'exclusive', 'scandal', 'bombshell', 'outrage',
    'expose', 'secret', 'revealed', 'banned', 'censored', 'deep state',
    'mainstream media', 'fake', 'hoax', 'conspiracy', 'miracle', 'cure',
    'destroyed', 'obliterated', 'wrecked', 'blasts', 'slams', 'rips',
    'explodes', 'erupts', 'unbelievable', 'stunning'
}

def clean_text(text: str, keep_sensational: bool = False) -> str:
    """
    Full text cleaning pipeline:
    1. Lowercase
    2. Remove URLs, emails, HTML tags
    3. Remove numbers and extra punctuation
    4. Remove stopwords
    5. Lemmatize
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    cleaned_tokens = []
    for token in tokens:
        if token not in STOP_WORDS and len(token) > 2:
            lemma = LEMMATIZER.lemmatize(token)
            cleaned_tokens.append(lemma)
    
    return ' '.join(cleaned_tokens)


def combine_title_text(title: str, text: str, title_weight: int = 3) -> str:
    """
    Combine title and text, repeating title for higher weight.
    Title is more indicative of fake news tone.
    """
    title_clean = clean_text(str(title))
    text_clean = clean_text(str(text))
    # Repeat title to give it more weight in TF-IDF
    combined = ' '.join([title_clean] * title_weight + [text_clean])
    return combined


def extract_features(text: str) -> dict:
    """
    Extract handcrafted features useful for explanation.
    Returns dict of feature values.
    """
    if not isinstance(text, str):
        return {}
    
    text_lower = text.lower()
    words = text.split()
    
    features = {
        'char_count': len(text),
        'word_count': len(words),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'sensational_word_count': sum(1 for w in SENSATIONAL_WORDS if w in text_lower),
        'number_count': len(re.findall(r'\d+', text)),
        'quote_count': text.count('"') + text.count("'"),
    }
    return features


def get_word_importance_scores(text: str, vectorizer, model) -> list:
    """
    Get per-word importance scores for text highlighting.
    Uses TF-IDF feature coefficients from logistic regression.
    Returns list of (word, score, is_important) tuples.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    words = text.split()
    result = []
    
    try:
        feature_names = vectorizer.get_feature_names_out()
        coef = model.coef_[0]  # shape: (n_features,)
        
        # Build word -> coefficient mapping
        word_coef = {}
        for fname, c in zip(feature_names, coef):
            word_coef[fname] = float(c)
        
        # Score each word in original text
        for word in words:
            clean_word = clean_text(word)
            score = word_coef.get(clean_word, 0.0)
            result.append({
                'word': word,
                'score': round(score, 4),
                'direction': 'fake' if score > 0.05 else ('real' if score < -0.05 else 'neutral')
            })
    except Exception:
        result = [{'word': w, 'score': 0.0, 'direction': 'neutral'} for w in words]
    
    return result

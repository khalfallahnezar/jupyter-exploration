"""
Text preprocessing pipeline — enhanced from the midterm NewsBot.
clean() produces a lightly-cleaned version (good for readability/sentiment);
preprocess() produces a fully-processed version (best for TF-IDF/classification).
"""
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

for _pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))


def clean(text):
    """Lightweight cleaning: strip markup/URLs/emails/symbols, lowercase, collapse whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(text, remove_stopwords=True, lemmatize=True, min_len=3):
    """Full pipeline: clean -> tokenize -> stop-word removal -> lemmatize -> length filter."""
    cleaned = clean(text)
    tokens = word_tokenize(cleaned)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _stop_words]
    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= min_len]
    return ' '.join(tokens)

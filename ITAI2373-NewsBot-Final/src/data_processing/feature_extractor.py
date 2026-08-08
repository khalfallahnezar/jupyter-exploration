"""
Shared feature-extraction helpers. The individual analysis/language modules
each fit their own vectorizer internally (so they stay independently
testable), but this module centralizes the custom, non-vectorizer features
used to enrich the midterm's classifier — kept here so any future model can
reuse them without duplicating the logic.
"""
import numpy as np


def length_features(texts):
    """Word count and character count — simple but useful signals that
    complement TF-IDF (e.g. sport articles run shorter than business ones)."""
    word_counts = np.array([len(t.split()) for t in texts]).reshape(-1, 1)
    char_counts = np.array([len(t) for t in texts]).reshape(-1, 1)
    return np.hstack([word_counts, char_counts])


def punctuation_density(texts):
    """Ratio of punctuation characters to total characters — a lightweight
    style signal (e.g. politics/business copy tends to use more commas and
    numerals than punchy sport headlines)."""
    import re
    out = []
    for t in texts:
        if not t:
            out.append(0.0)
            continue
        punct = len(re.findall(r'[,.;:!?]', t))
        out.append(punct / max(len(t), 1))
    return np.array(out).reshape(-1, 1)

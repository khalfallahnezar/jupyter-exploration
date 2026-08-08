"""
NewsBot 2.0 - Module A: Advanced Content Analysis Engine
AdvancedNewsClassifier, TopicDiscoveryEngine, SentimentEvolutionTracker, EntityRelationshipMapper
"""
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import LatentDirichletAllocation, NMF

import networkx as nx


# ---------------------------------------------------------------------------
class AdvancedNewsClassifier:
    """
    Enhanced news classification with confidence scoring and multi-label support.
    Builds on the midterm's TF-IDF + Logistic Regression classifier, but exposes
    calibrated probabilities so low-confidence articles can be routed to a human
    reviewer instead of being force-classified into a single bucket.
    """

    def __init__(self, max_features=5000, multi_label_threshold=0.20):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                           stop_words='english', min_df=2)
        self.model = LogisticRegression(max_iter=1000, C=2.0)
        self.classes_ = None
        self.multi_label_threshold = multi_label_threshold
        self.is_fitted = False

    def fit(self, texts, labels):
        """Fit on a train split and report held-out test performance, then
        refit on the full dataset so the deployed model sees all available
        data (same two-stage pattern as the midterm classifier)."""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels)

        Xtr = self.vectorizer.fit_transform(X_train)
        self.model.fit(Xtr, y_train)
        Xte = self.vectorizer.transform(X_test)
        preds = self.model.predict(Xte)
        self.holdout_report_ = {
            'accuracy': round(accuracy_score(y_test, preds), 4),
            'macro_f1': round(f1_score(y_test, preds, average='macro'), 4),
            'report': classification_report(y_test, preds, output_dict=False),
            'n_test': len(y_test),
        }

        # Refit on the full corpus for the model actually used in production.
        X_full = self.vectorizer.fit_transform(texts)
        self.model.fit(X_full, labels)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def evaluate(self, texts, labels, cv=5):
        """Held-out test performance from fit(), plus a fresh k-fold
        cross-validation score for stability — mirrors the midterm's
        train/test + cross-validation reporting."""
        X_full = self.vectorizer.transform(texts)
        cv_scores = cross_val_score(self.model, X_full, labels, cv=cv, scoring='f1_macro')
        return {
            'accuracy': self.holdout_report_['accuracy'],
            'macro_f1': self.holdout_report_['macro_f1'],
            'report': self.holdout_report_['report'],
            'n_test': self.holdout_report_['n_test'],
            'cv_macro_f1_mean': round(float(cv_scores.mean()), 4),
            'cv_macro_f1_std': round(float(cv_scores.std()), 4),
        }

    def predict(self, text):
        """Single-label prediction with a calibrated confidence score."""
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        top_idx = int(np.argmax(proba))
        return {
            'category': self.classes_[top_idx],
            'confidence': round(float(proba[top_idx]), 4),
            'needs_human_review': bool(proba[top_idx] < 0.55),
        }

    def predict_multi_label(self, text):
        """Return every category whose probability clears the multi-label threshold,
        for articles that genuinely straddle two desks (e.g. business + tech)."""
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        labels = [
            {'category': self.classes_[i], 'confidence': round(float(p), 4)}
            for i, p in enumerate(proba) if p >= self.multi_label_threshold
        ]
        return sorted(labels, key=lambda d: -d['confidence'])


# ---------------------------------------------------------------------------
class TopicDiscoveryEngine:
    """
    Advanced topic modeling for discovering themes and trends across the corpus,
    using either LDA (probabilistic) or NMF (parts-based, often crisper topics
    on shorter news text).
    """

    def __init__(self, n_topics=10, method='lda', max_features=3000):
        self.n_topics = n_topics
        self.method = method
        if method == 'lda':
            self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english',
                                               min_df=3, max_df=0.9)
            self.model = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                                     learning_method='online', max_iter=10)
        else:
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english',
                                               min_df=3, max_df=0.9)
            self.model = NMF(n_components=n_topics, random_state=42, init='nndsvda', max_iter=400)
        self.doc_topic_matrix = None
        self.feature_names = None

    def fit_transform(self, documents):
        X = self.vectorizer.fit_transform(documents)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.doc_topic_matrix = self.model.fit_transform(X)
        return self.doc_topic_matrix

    def get_topic_words(self, topic_id, n_words=10):
        weights = self.model.components_[topic_id]
        top_idx = weights.argsort()[::-1][:n_words]
        return list(self.feature_names[top_idx])

    def get_all_topics(self, n_words=8):
        return {i: self.get_topic_words(i, n_words) for i in range(self.n_topics)}

    def get_document_topics(self, doc_index, top_n=3):
        dist = self.doc_topic_matrix[doc_index]
        top_idx = dist.argsort()[::-1][:top_n]
        return [{'topic': int(i), 'weight': round(float(dist[i]), 4),
                  'top_words': self.get_topic_words(i, 5)} for i in top_idx]

    def topics_by_category(self, category_series):
        """Which discovered topics dominate each known category — a proxy for
        'topic evolution' since the BBC dataset has no publish dates."""
        dominant_topic = self.doc_topic_matrix.argmax(axis=1)
        df = pd.DataFrame({'category': category_series.values, 'topic': dominant_topic})
        return pd.crosstab(df['category'], df['topic'])


# ---------------------------------------------------------------------------
class SentimentEvolutionTracker:
    """
    Advanced sentiment analysis with tracking across a sequence of articles.
    Note: the BBC dataset has no publish dates, so 'evolution' here is tracked
    across document order / category rather than true wall-clock time — the
    same rolling-window technique applies directly once real timestamps exist.
    """

    def __init__(self, window=25):
        from nltk.sentiment import SentimentIntensityAnalyzer
        self.sia = SentimentIntensityAnalyzer()
        self.window = window

    def score(self, text):
        s = self.sia.polarity_scores(text)
        label = 'Positive' if s['compound'] >= 0.05 else 'Negative' if s['compound'] <= -0.05 else 'Neutral'
        return {'compound': s['compound'], 'label': label, 'pos': s['pos'], 'neg': s['neg'], 'neu': s['neu']}

    def score_corpus(self, texts):
        return [self.score(t)['compound'] for t in texts]

    def rolling_trend(self, texts):
        scores = pd.Series(self.score_corpus(texts))
        return scores.rolling(window=self.window, min_periods=1).mean()

    def evolution_by_category(self, df, text_col='content', category_col='category'):
        out = {}
        for cat, group in df.groupby(category_col):
            out[cat] = self.rolling_trend(group[text_col].tolist()).tolist()
        return out


# ---------------------------------------------------------------------------
class EntityRelationshipMapper:
    """
    Advanced NER with relationship extraction: entities that co-occur in the
    same article are linked in a graph, so we can see which people/orgs/places
    tend to appear together across the corpus (a lightweight relationship map,
    without needing a dedicated relation-extraction model).
    """

    def __init__(self, spacy_model):
        self.nlp = spacy_model
        self.graph = nx.Graph()

    def extract_entities(self, text, labels=('PERSON', 'ORG', 'GPE')):
        doc = self.nlp(text)
        seen = []
        for ent in doc.ents:
            if ent.label_ in labels and len(ent.text.strip()) > 1:
                seen.append((ent.text.strip(), ent.label_))
        # de-dupe while keeping order
        out, seen_texts = [], set()
        for text_, label_ in seen:
            key = text_.lower()
            if key not in seen_texts:
                seen_texts.add(key)
                out.append({'text': text_, 'label': label_})
        return out

    def build_graph(self, texts, max_docs=None):
        docs = texts if max_docs is None else texts[:max_docs]
        for text in docs:
            ents = [e['text'] for e in self.extract_entities(text)]
            for e in ents:
                self.graph.add_node(e)
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    if self.graph.has_edge(ents[i], ents[j]):
                        self.graph[ents[i]][ents[j]]['weight'] += 1
                    else:
                        self.graph.add_edge(ents[i], ents[j], weight=1)
        return self.graph

    def top_entities(self, n=15):
        centrality = nx.degree_centrality(self.graph)
        return sorted(centrality.items(), key=lambda x: -x[1])[:n]

    def related_entities(self, entity, n=5):
        if entity not in self.graph:
            return []
        neighbors = sorted(self.graph[entity].items(), key=lambda x: -x[1]['weight'])[:n]
        return [{'entity': nb, 'co_occurrences': d['weight']} for nb, d in neighbors]

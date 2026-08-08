"""
NewsBot 2.0 - Module B: Language Understanding & Generation
IntelligentSummarizer, SemanticSearchEngine, ContentEnhancer
"""
import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize


# ---------------------------------------------------------------------------
class IntelligentSummarizer:
    """
    Extractive summarization using a TextRank-style algorithm: sentences are
    embedded with TF-IDF, a sentence-similarity graph is built, and PageRank
    ranks sentences by how well they represent the whole article. This avoids
    a heavyweight downloaded model while still being a legitimate, well-known
    summarization technique (Mihalcea & Tarau, 2004).
    """

    def __init__(self, ratio=0.3, min_sentences=2, max_sentences=6):
        self.ratio = ratio
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def summarize(self, text):
        sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 0]
        if len(sentences) <= self.min_sentences:
            return {'summary': text.strip(), 'compression_ratio': 1.0, 'n_sentences': len(sentences)}

        vec = TfidfVectorizer(stop_words='english')
        try:
            X = vec.fit_transform(sentences)
        except ValueError:
            return {'summary': text.strip(), 'compression_ratio': 1.0, 'n_sentences': len(sentences)}

        sim_matrix = cosine_similarity(X)
        np.fill_diagonal(sim_matrix, 0)
        graph = nx.from_numpy_array(sim_matrix)
        try:
            scores = nx.pagerank(graph, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            scores = {i: 1.0 for i in range(len(sentences))}

        n_target = max(self.min_sentences, min(self.max_sentences, round(len(sentences) * self.ratio)))
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:n_target]
        keep_idx = sorted(i for i, _ in ranked)
        summary = ' '.join(sentences[i] for i in keep_idx)
        return {
            'summary': summary,
            'compression_ratio': round(len(summary) / max(len(text), 1), 3),
            'n_sentences': len(keep_idx),
        }


# ---------------------------------------------------------------------------
class SemanticSearchEngine:
    """
    Semantic search using TF-IDF + Truncated SVD (Latent Semantic Analysis) to
    produce dense 'semantic embeddings' for every document, then ranks by
    cosine similarity. LSA is chosen over a downloaded transformer model so the
    notebook has no external model-hub dependency and stays reproducible
    offline; swapping in sentence-transformers is a drop-in upgrade (see docs).
    """

    def __init__(self, n_components=100):
        self.vectorizer = TfidfVectorizer(max_features=8000, stop_words='english', min_df=2)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.doc_vectors = None
        self.documents = None

    def fit(self, documents):
        self.documents = list(documents)
        X = self.vectorizer.fit_transform(self.documents)
        n_comp = min(self.svd.n_components, X.shape[1] - 1, X.shape[0] - 1)
        if n_comp != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=max(2, n_comp), random_state=42)
        self.doc_vectors = self.svd.fit_transform(X)
        return self

    def _embed_query(self, query):
        Xq = self.vectorizer.transform([query])
        return self.svd.transform(Xq)

    def search(self, query, top_k=5):
        qv = self._embed_query(query)
        sims = cosine_similarity(qv, self.doc_vectors)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [
            {'doc_index': int(i), 'similarity': round(float(sims[i]), 4),
             'preview': self.documents[i][:180] + '...'}
            for i in top_idx
        ]

    def most_similar_to_doc(self, doc_index, top_k=5):
        sims = cosine_similarity([self.doc_vectors[doc_index]], self.doc_vectors)[0]
        sims[doc_index] = -1  # exclude itself
        top_idx = sims.argsort()[::-1][:top_k]
        return [{'doc_index': int(i), 'similarity': round(float(sims[i]), 4)} for i in top_idx]


# ---------------------------------------------------------------------------
class ContentEnhancer:
    """
    Advanced content analysis and enhancement: turns the raw outputs of the
    other modules (classification, sentiment, entities, topics) into a short,
    human-readable "intelligence brief" plus a keyword list and related-reading
    suggestions.
    """

    def __init__(self, semantic_search_engine=None):
        self.search_engine = semantic_search_engine

    def key_phrases(self, text, n=8):
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2000)
        try:
            X = vec.fit_transform([text])
        except ValueError:
            return []
        scores = X.toarray()[0]
        terms = np.array(vec.get_feature_names_out())
        top_idx = scores.argsort()[::-1][:n]
        return [t for t in terms[top_idx] if scores[terms.tolist().index(t)] > 0]

    def generate_brief(self, text, classification, sentiment, entities, topics=None, doc_index=None):
        lines = []
        lines.append(f"Classified as **{classification['category']}** "
                      f"(confidence {classification['confidence']:.0%}"
                      f"{', flagged for human review' if classification['needs_human_review'] else ''}).")
        lines.append(f"Overall tone is **{sentiment['label']}** (compound score {sentiment['compound']:.2f}).")
        if entities:
            top_ents = ', '.join(e['text'] for e in entities[:5])
            lines.append(f"Key entities mentioned: {top_ents}.")
        if topics:
            top_words = topics[0]['top_words'][:5]
            lines.append(f"Dominant theme relates to: {', '.join(top_words)}.")
        related = []
        if self.search_engine is not None and doc_index is not None:
            related = self.search_engine.most_similar_to_doc(doc_index, top_k=3)
        return {
            'brief': ' '.join(lines),
            'key_phrases': self.key_phrases(text),
            'related_articles': related,
        }

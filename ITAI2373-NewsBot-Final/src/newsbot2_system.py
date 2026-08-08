"""
NewsBot 2.0 - System Integration & Evaluation
NewsBot2Config, NewsBot2IntegratedSystem, NewsBot2TestSuite, NewsBot2Evaluator
"""
import time
import numpy as np
import pandas as pd

from src.analysis.content_analysis import (AdvancedNewsClassifier, TopicDiscoveryEngine,
                                            SentimentEvolutionTracker, EntityRelationshipMapper)
from src.language_models.language_understanding import (IntelligentSummarizer,
                                                          SemanticSearchEngine, ContentEnhancer)
from src.multilingual.multilingual_processor import MultilingualProcessor
from src.conversation.conversational_interface import ConversationalInterface


class NewsBot2Config:
    """Central configuration for every NewsBot 2.0 component."""

    def __init__(self, n_topics=8, topic_method='lda', classifier_max_features=5000,
                 semantic_dims=100, entity_labels=('PERSON', 'ORG', 'GPE'),
                 multi_label_threshold=0.20, summarizer_ratio=0.3):
        self.n_topics = n_topics
        self.topic_method = topic_method
        self.classifier_max_features = classifier_max_features
        self.semantic_dims = semantic_dims
        self.entity_labels = entity_labels
        self.multi_label_threshold = multi_label_threshold
        self.summarizer_ratio = summarizer_ratio


class NewsBot2IntegratedSystem:
    """
    Complete NewsBot 2.0 system: wires together classification, topic
    modeling, sentiment tracking, entity relationships, summarization,
    semantic search, content enhancement, multilingual processing, and the
    conversational interface into one object with a simple public API.
    """

    def __init__(self, config: NewsBot2Config, spacy_model):
        self.config = config
        self.spacy_model = spacy_model
        self.df = None

        self.classifier = AdvancedNewsClassifier(max_features=config.classifier_max_features,
                                                   multi_label_threshold=config.multi_label_threshold)
        self.topic_engine = TopicDiscoveryEngine(n_topics=config.n_topics, method=config.topic_method)
        self.sentiment_tracker = SentimentEvolutionTracker()
        self.entity_mapper = EntityRelationshipMapper(spacy_model)
        self.summarizer = IntelligentSummarizer(ratio=config.summarizer_ratio)
        self.search_engine = SemanticSearchEngine(n_components=config.semantic_dims)
        self.content_enhancer = ContentEnhancer(semantic_search_engine=self.search_engine)
        self.multilingual = MultilingualProcessor()
        self.conversation = ConversationalInterface(self)

        self._is_trained = False
        self.training_report = {}

    # ------------------------------------------------------------------
    def train(self, df, text_col='content', category_col='category', entity_sample_size=300):
        """Fit every learned component on the corpus. Entity graph is built on
        a sample for speed — this is the main runtime cost in a large corpus."""
        t0 = time.time()
        self.df = df.reset_index(drop=True)
        texts = self.df[text_col].tolist()
        labels = self.df[category_col].tolist()

        self.classifier.fit(texts, labels)
        clf_eval = self.classifier.evaluate(texts, labels)

        self.topic_engine.fit_transform(texts)
        self.search_engine.fit(texts)
        self.entity_mapper.build_graph(texts, max_docs=entity_sample_size)

        # pre-compute sentiment for the whole corpus once (reused everywhere)
        self.df['sentiment_compound'] = self.sentiment_tracker.score_corpus(texts)
        self.df['sentiment_label'] = self.df['sentiment_compound'].apply(
            lambda c: 'Positive' if c >= 0.05 else 'Negative' if c <= -0.05 else 'Neutral')

        self._is_trained = True
        self.training_report = {
            'n_documents': len(self.df),
            'classifier_accuracy': round(clf_eval['accuracy'], 4),
            'classifier_macro_f1': round(clf_eval['macro_f1'], 4),
            'n_topics': self.config.n_topics,
            'n_entities_in_graph': self.entity_mapper.graph.number_of_nodes(),
            'training_time_sec': round(time.time() - t0, 1),
        }
        return self.training_report

    # ------------------------------------------------------------------
    def analyze_article(self, text):
        """Full-pipeline analysis of a single (possibly new/unseen) article."""
        if not self._is_trained:
            raise RuntimeError("Call .train(df) before analyzing articles.")

        lang = self.multilingual.detect_language(text)
        working_text = text
        if lang['language_code'] != 'en':
            translation = self.multilingual.translate_to_english(text)
            working_text = translation['translated_text']

        classification = self.classifier.predict(working_text)
        multi_label = self.classifier.predict_multi_label(working_text)
        sentiment = self.sentiment_tracker.score(working_text)
        entities = self.entity_mapper.extract_entities(working_text, labels=self.config.entity_labels)
        summary = self.summarizer.summarize(working_text)
        enhancement = self.content_enhancer.generate_brief(
            working_text, classification, sentiment, entities)

        return {
            'language': lang,
            'classification': classification,
            'multi_label_categories': multi_label,
            'sentiment': sentiment,
            'entities': entities,
            'summary': summary,
            'enhancement': enhancement,
        }

    # ------------------------------------------------------------------
    def search(self, query, top_k=5):
        return self.search_engine.search(query, top_k=top_k)

    def filter_articles(self, category=None, sentiment=None):
        df = self.df
        if category:
            df = df[df['category'].str.lower() == category.lower()]
        if sentiment:
            df = df[df['sentiment_label'] == sentiment]
        return df

    def ask(self, query):
        return self.conversation.process(query)


class NewsBot2TestSuite:
    """Lightweight sanity-check tests for every NewsBot 2.0 component. Not a
    replacement for pytest, but gives an at-a-glance pass/fail readout that
    can run inside the notebook (see tests/ for the pytest equivalents)."""

    def __init__(self, newsbot_system: NewsBot2IntegratedSystem):
        self.newsbot = newsbot_system
        self.results = []

    def _check(self, name, condition):
        self.results.append({'test': name, 'passed': bool(condition)})

    def run_all(self):
        self.results = []
        s = self.newsbot
        sample_text = s.df.iloc[0]['content']

        self._check('system_is_trained', s._is_trained)
        self._check('classifier_predicts_known_category',
                     s.classifier.predict(sample_text)['category'] in s.classifier.classes_)
        self._check('topic_engine_has_topics', s.topic_engine.doc_topic_matrix is not None
                     and s.topic_engine.doc_topic_matrix.shape[1] == s.config.n_topics)
        self._check('sentiment_score_in_range',
                     -1.0 <= s.sentiment_tracker.score(sample_text)['compound'] <= 1.0)
        self._check('entity_graph_nonempty', s.entity_mapper.graph.number_of_nodes() > 0)
        self._check('summarizer_shorter_than_source',
                     len(s.summarizer.summarize(sample_text)['summary']) <= len(sample_text) + 5)
        self._check('semantic_search_returns_results', len(s.search('news', top_k=3)) > 0)
        self._check('multilingual_detects_english',
                     s.multilingual.detect_language(sample_text)['language_code'] == 'en')
        self._check('conversation_handles_count_query',
                     'response' in s.ask('how many sport articles are there'))
        self._check('analyze_article_end_to_end',
                     'classification' in s.analyze_article(sample_text[:500]))

        n_pass = sum(r['passed'] for r in self.results)
        return {'passed': n_pass, 'total': len(self.results), 'details': self.results}


class NewsBot2Evaluator:
    """Corpus-level evaluation metrics used in the technical documentation
    and executive summary."""

    def __init__(self, newsbot_system: NewsBot2IntegratedSystem):
        self.newsbot = newsbot_system

    def evaluate_classifier(self):
        s = self.newsbot
        return s.classifier.evaluate(s.df['content'].tolist(), s.df['category'].tolist())

    def evaluate_topics(self):
        """Topic 'coherence' proxy: how distinct each topic's top words are
        from the other topics (low overlap = more distinct, interpretable
        topics)."""
        s = self.newsbot
        all_topics = s.topic_engine.get_all_topics(n_words=10)
        overlaps = []
        topic_ids = list(all_topics.keys())
        for i in range(len(topic_ids)):
            for j in range(i + 1, len(topic_ids)):
                a, b = set(all_topics[topic_ids[i]]), set(all_topics[topic_ids[j]])
                overlaps.append(len(a & b) / len(a | b))
        return {'n_topics': s.config.n_topics, 'avg_topic_overlap': round(float(np.mean(overlaps)), 3),
                'topics': all_topics}

    def evaluate_summarizer(self, n_samples=30):
        s = self.newsbot
        sample = s.df['content'].sample(min(n_samples, len(s.df)), random_state=42)
        ratios = [s.summarizer.summarize(t)['compression_ratio'] for t in sample]
        return {'n_samples': len(sample), 'avg_compression_ratio': round(float(np.mean(ratios)), 3)}

    def full_report(self):
        return {
            'training_report': self.newsbot.training_report,
            'classifier': self.evaluate_classifier(),
            'topics': self.evaluate_topics(),
            'summarizer': self.evaluate_summarizer(),
        }

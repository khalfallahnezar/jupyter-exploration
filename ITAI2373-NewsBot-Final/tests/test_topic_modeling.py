import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.analysis.content_analysis import TopicDiscoveryEngine

DOCS = (
    ["the stock market and shares and earnings report for the company"] * 10 +
    ["the football match and team and player and goal in the league"] * 10
)


def test_lda_fit_transform_shape():
    engine = TopicDiscoveryEngine(n_topics=3, method='lda')
    matrix = engine.fit_transform(DOCS)
    assert matrix.shape == (len(DOCS), 3)


def test_nmf_fit_transform_shape():
    engine = TopicDiscoveryEngine(n_topics=3, method='nmf')
    matrix = engine.fit_transform(DOCS)
    assert matrix.shape == (len(DOCS), 3)


def test_get_topic_words_returns_words():
    engine = TopicDiscoveryEngine(n_topics=3, method='lda')
    engine.fit_transform(DOCS)
    words = engine.get_topic_words(0, n_words=5)
    assert len(words) == 5
    assert all(isinstance(w, str) for w in words)


def test_get_document_topics_returns_topN():
    engine = TopicDiscoveryEngine(n_topics=3, method='lda')
    engine.fit_transform(DOCS)
    topics = engine.get_document_topics(0, top_n=2)
    assert len(topics) == 2
    assert all('topic' in t and 'weight' in t for t in topics)

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False

from src.newsbot2_system import NewsBot2Config, NewsBot2IntegratedSystem, NewsBot2TestSuite

DOCS = {
    'content': (
        ["Apple Inc reported record earnings and its shares rose sharply in New York today."] * 8 +
        ["Manchester United won the championship match against Chelsea with a stunning goal."] * 8 +
        ["The White House announced a new policy after the election results in Washington."] * 8
    ),
    'category': (["business"] * 8 + ["sport"] * 8 + ["politics"] * 8),
}


@pytest.mark.skipif(not _SPACY_OK, reason="spaCy en_core_web_sm not installed")
def test_full_system_trains_and_passes_internal_suite():
    df = pd.DataFrame(DOCS)
    config = NewsBot2Config(n_topics=3, classifier_max_features=200)
    system = NewsBot2IntegratedSystem(config, _nlp)
    system.train(df, entity_sample_size=24)

    suite = NewsBot2TestSuite(system)
    results = suite.run_all()
    assert results['passed'] == results['total']


@pytest.mark.skipif(not _SPACY_OK, reason="spaCy en_core_web_sm not installed")
def test_analyze_article_returns_all_sections():
    df = pd.DataFrame(DOCS)
    config = NewsBot2Config(n_topics=3, classifier_max_features=200)
    system = NewsBot2IntegratedSystem(config, _nlp)
    system.train(df, entity_sample_size=24)

    result = system.analyze_article("Shares in the company jumped after a strong earnings report.")
    for key in ('language', 'classification', 'sentiment', 'entities', 'summary', 'enhancement'):
        assert key in result

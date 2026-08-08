import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data_processing.text_preprocessor import clean, preprocess
from src.data_processing.data_validator import validate_corpus
import pandas as pd


def test_clean_lowercases_and_strips_punctuation():
    assert clean("Hello, WORLD!!") == "hello world"


def test_clean_handles_nan():
    assert clean(float('nan')) == ""


def test_preprocess_removes_stopwords_and_short_tokens():
    out = preprocess("The quick fox is running to a big house")
    assert "the" not in out.split()
    assert "quick" in out or "fox" in out


def test_validate_corpus_flags_missing_text():
    df = pd.DataFrame({'content': ['a valid article here', None], 'category': ['sport', 'tech']})
    result = validate_corpus(df, min_articles=1, min_categories=1)
    assert not result['is_valid']
    assert any('missing text' in issue for issue in result['issues'])


def test_validate_corpus_passes_clean_data():
    df = pd.DataFrame({
        'content': [f"article number {i} about business news" for i in range(10)],
        'category': ['business'] * 5 + ['tech'] * 5,
    })
    result = validate_corpus(df, min_articles=5, min_categories=2)
    assert result['is_valid']

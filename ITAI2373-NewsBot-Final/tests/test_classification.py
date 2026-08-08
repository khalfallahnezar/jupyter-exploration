import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.analysis.content_analysis import AdvancedNewsClassifier

TRAIN_TEXTS = (
    ["stock market shares rose today as the company reported earnings"] * 15 +
    ["the football team won the match with a late goal"] * 15 +
    ["the election campaign focused on new government policy"] * 15
)
TRAIN_LABELS = (["business"] * 15 + ["sport"] * 15 + ["politics"] * 15)


def _fitted_classifier():
    clf = AdvancedNewsClassifier(max_features=500)
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    return clf


def test_classifier_fits_without_error():
    clf = _fitted_classifier()
    assert clf.is_fitted


def test_classifier_predicts_known_category():
    clf = _fitted_classifier()
    result = clf.predict("the company's shares rallied after strong earnings")
    assert result['category'] in clf.classes_
    assert 0.0 <= result['confidence'] <= 1.0


def test_classifier_multi_label_returns_list():
    clf = _fitted_classifier()
    labels = clf.predict_multi_label("the match ended and shares in the club rose")
    assert isinstance(labels, list)
    assert all('category' in l and 'confidence' in l for l in labels)


def test_evaluate_reports_holdout_metrics():
    clf = _fitted_classifier()
    result = clf.evaluate(TRAIN_TEXTS, TRAIN_LABELS, cv=3)
    assert 0.0 <= result['accuracy'] <= 1.0
    assert 0.0 <= result['macro_f1'] <= 1.0
    assert 'cv_macro_f1_mean' in result

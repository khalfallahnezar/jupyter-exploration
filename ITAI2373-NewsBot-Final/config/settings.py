"""
Configuration management for NewsBot 2.0.
Import NewsBot2Config from src.newsbot2_system for the runtime object; this
file holds the default values and any environment-specific overrides.
"""

DATA_URL = ("https://raw.githubusercontent.com/suraj-deshmukh/"
            "BBC-Dataset-News-Classification/master/dataset/dataset.csv")

DEFAULT_N_TOPICS = 8
DEFAULT_TOPIC_METHOD = "lda"          # "lda" or "nmf"
DEFAULT_CLASSIFIER_MAX_FEATURES = 5000
DEFAULT_SEMANTIC_DIMS = 100
DEFAULT_ENTITY_LABELS = ("PERSON", "ORG", "GPE")
DEFAULT_MULTI_LABEL_THRESHOLD = 0.20
DEFAULT_SUMMARIZER_RATIO = 0.3

SPACY_MODEL = "en_core_web_sm"

# Translation is optional and calls an external service (deep-translator /
# Google Translate). If TRANSLATION_ENABLED is False, MultilingualProcessor
# will only detect language and skip translation entirely.
TRANSLATION_ENABLED = True

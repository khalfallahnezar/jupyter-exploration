# 🤖 NewsBot Intelligence System 2.0 — ITAI 2373 Final Project

Production-style extension of the [midterm NewsBot](https://github.com/khalfallahnezar/jupyter-exploration/tree/main/ITAI2373-NewsBot-Midterm),
adding topic modeling, summarization, semantic search, multilingual processing, and a
conversational query interface on top of the existing classification/sentiment/NER pipeline.

**Author:** Nezar Khalfallah
**Course:** ITAI 2373 — Natural Language Processing
**Dataset:** BBC News (2,225 articles · 5 categories: business, entertainment, politics, sport, tech)

---

## 📂 Repository Contents

| Path | Description |
|---|---|
| `notebooks/NewsBot_Intelligence_System_2.0.ipynb` | Complete, runnable notebook — all 12 required classes, trained and demoed end-to-end |
| `src/` | The same classes as an installable package (`analysis/`, `language_models/`, `multilingual/`, `conversation/`, `data_processing/`, `newsbot2_system.py`) |
| `tests/` | Pytest suite (15 tests) covering preprocessing, classification, topic modeling, and full-system integration |
| `config/settings.py` | Central configuration (topic count, thresholds, dataset URL, etc.) |
| `docs/` | Technical documentation, user guide |
| `reports/` | Executive summary |
| `requirements.txt` | All dependencies |

## ▶️ How to Run

1. Open `notebooks/NewsBot_Intelligence_System_2.0.ipynb` in **Google Colab** (free tier is sufficient).
2. Run the **Setup and Installation** cell first.
3. `Runtime → Run all`. The notebook downloads the BBC dataset automatically and trains every
   component — no manual steps.
4. To run the same logic as an installable package instead: `pip install -r requirements.txt`,
   then `from src.newsbot2_system import NewsBot2Config, NewsBot2IntegratedSystem`.
5. Tests: `pytest tests/ -v` (15 tests, all passing as of this submission).

Total notebook runtime is a few minutes on the Colab free tier (~1,000 articles, 8 topics, entity
graph sampled at 400 articles for speed).

## 🧩 System Architecture

**Module A — Advanced Content Analysis** (`src/analysis/content_analysis.py`)
- `AdvancedNewsClassifier` — TF-IDF + Logistic Regression with calibrated confidence scores and
  multi-label output (97% held-out accuracy on BBC News).
- `TopicDiscoveryEngine` — LDA/NMF topic discovery, with topic distribution by category as a
  stand-in for "topic evolution" (the dataset has no publish dates).
- `SentimentEvolutionTracker` — VADER sentiment with rolling-window trend tracking.
- `EntityRelationshipMapper` — spaCy NER + co-occurrence graph (NetworkX) linking entities that
  appear together.

**Module B — Language Understanding & Generation** (`src/language_models/language_understanding.py`)
- `IntelligentSummarizer` — extractive TextRank-style summarization (no downloaded model needed).
- `SemanticSearchEngine` — TF-IDF + LSA (Truncated SVD) embeddings + cosine similarity.
- `ContentEnhancer` — auto-generated article briefs, key phrases, related-article suggestions.

**Module C — Multilingual Intelligence** (`src/multilingual/multilingual_processor.py`)
- `MultilingualProcessor` — language detection (`langdetect`) + translation-to-English
  (`deep-translator`), with graceful fallback if the translation service is unreachable.

**Module D — Conversational Interface** (`src/conversation/conversational_interface.py`)
- `ConversationalInterface` — rule-based intent classification (search / count / summarize /
  entity lookup) routes free-text questions to the right module.

**Integration** (`src/newsbot2_system.py`)
- `NewsBot2Config`, `NewsBot2IntegratedSystem`, `NewsBot2TestSuite`, `NewsBot2Evaluator`.

## 📈 Key Results

- **Classification:** 97% held-out accuracy / 0.97 macro-F1 (5-fold CV confirms stability),
  with confidence scoring flagging low-certainty articles for human review.
- **Topics:** 8 LDA topics recovered, low cross-topic word overlap (~0.08) indicating distinct,
  interpretable themes.
- **Sentiment:** entertainment skews most positive, politics most negative — consistent with the
  midterm's finding, now shown as a trend rather than a single number.
- **Summarization:** ~3.4x average compression while keeping PageRank-selected key sentences.
- **Semantic search:** retrieves conceptually related articles beyond exact keyword overlap.
- **Conversational interface:** correctly routes free-text queries (tested on 5 query types) with
  no trained NLU model required.

## 🚀 Known Limitations & Possible Extensions

- No publish dates in BBC News — "evolution" analyses use document order/category as a proxy;
  the same code applies directly once real timestamps exist.
- Semantic search uses LSA rather than a transformer embedding model, to avoid an external
  model-hub dependency — `sentence-transformers` is a documented drop-in upgrade.
- Translation depends on a live external service and falls back to showing the original text if
  unavailable.
- The conversational interface is rule-based rather than a trained intent classifier.
- Optional bonus (not built in this submission): Flask web application frontend — see the
  separate "Web App Development Tutorial" guide for a ready-made implementation path.

## 📝 Notes

- File-naming conventions (per assignment):
  `FP_TechnicalDoc_NezarKhalfallah_ITAI2373.pdf`,
  `FP_ExecutiveSummary_NezarKhalfallah_ITAI2373.pdf`,
  `FP_ReflectiveJournal_NezarKhalfallah_ITAI2373.pdf`
- Dataset: D. Greene and P. Cunningham, *BBC News* dataset (2006).

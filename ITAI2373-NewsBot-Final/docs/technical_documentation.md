# NewsBot Intelligence System 2.0 — Technical Documentation
Nezar Khalfallah | ITAI 2373 — Natural Language Processing | Final Project

## 1. Architecture Overview

NewsBot 2.0 extends the midterm's single-notebook classification/sentiment/NER pipeline into a
modular system with four capability areas, each implemented as an independently testable class,
orchestrated by one integration layer.



NewsBot2IntegratedSystem
                            


Every component reads from one canonical `content` column and shares a single trained corpus
(`system.df`), the same integration discipline used in the midterm project — components consume
one canonical data representation instead of each rebuilding their own inputs.

`NewsBot2Config` centralizes every tunable parameter (topic count, classifier feature cap,
semantic embedding dimensions, entity label set, multi-label threshold, summarization ratio) so
the whole system can be re-tuned from one place without touching component code.

## 2. Component Design & Key Decisions

### 2.1 Advanced Content Analysis (Module A)

**AdvancedNewsClassifier** — TF-IDF (unigrams + bigrams, 5,000 features) feeding a Logistic
Regression classifier. Two changes from the midterm's classifier:

- `predict()` returns a calibrated confidence score (`predict_proba`) and a
  `needs_human_review` flag when confidence falls below 55% — turning a black-box prediction
  into an auditable routing decision, which is how classification is actually used in a
  media-monitoring pipeline.
- `predict_multi_label()` returns every category clearing a configurable probability threshold
  (default 20%), for articles that legitimately span two desks (e.g. a story about a tech
  company's stock price is both "tech" and "business").
- `fit()` trains on an 80/20 stratified split, reports held-out performance, then refits on the
  full corpus for the deployed model — the same two-stage pattern used in the midterm, which
  avoids reporting inflated training-set accuracy as if it were generalization performance.

**TopicDiscoveryEngine** — wraps both LDA (`CountVectorizer` + `LatentDirichletAllocation`)
and NMF (`TfidfVectorizer` + `NMF`) behind one interface, selectable via `NewsBot2Config`.
LDA is the default because it handles the modest corpus size well and produces interpretable
per-topic word distributions. Because the BBC dataset carries no publish dates, "topic evolution
over time" is implemented as topic distribution *by category* (`topics_by_category`) — a
defensible proxy that exercises the identical code path a dated corpus would use.

**SentimentEvolutionTracker** — reuses the midterm's VADER scorer, adding a rolling-window
mean (`rolling_trend`) and a per-category breakdown (`evolution_by_category`). Same
timestamp limitation as above: the "evolution" axis is document order within a category rather
than wall-clock time.

**EntityRelationshipMapper** — spaCy NER restricted to PERSON/ORG/GPE, with entities that
co-occur in the same article linked in a NetworkX graph (edge weight = co-occurrence count).
`top_entities()` ranks by degree centrality (structurally central to the corpus);
`related_entities()` answers "who is most associated with X?" — directly supporting the
conversational interface's `entity_lookup` intent. Built on a capped sample (400 articles) since
this is the most expensive step per document; the cap is a `NewsBot2Config`-adjacent parameter
(`entity_sample_size` on `.train()`).

### 2.2 Language Understanding & Generation (Module B)

**IntelligentSummarizer** — extractive summarization via TextRank (Mihalcea & Tarau, 2004):
sentences are embedded with TF-IDF, a sentence-similarity graph is built, and PageRank ranks
sentences by centrality; the top-ranked sentences (by original order) form the summary. This was
chosen over an abstractive transformer model (e.g. BART/T5) for two reasons: it needs no
downloaded model weights (fully reproducible offline), and extractive summaries cannot
hallucinate facts that were never in the source article — an important property for a news
system.

**SemanticSearchEngine** — TF-IDF (8,000 features) reduced to 100 dimensions via Truncated SVD
(Latent Semantic Analysis), ranked by cosine similarity. LSA was chosen over
`sentence-transformers` to avoid an external model-hub download dependency in a Colab
free-tier / grading context; swapping the embedding step for a transformer encoder requires no
change to `.search()` or `.most_similar_to_doc()`, since both operate purely on the resulting
vector space.

**ContentEnhancer** — assembles a human-readable brief from the outputs of the other
components (classification, sentiment, entities, topics), plus TF-IDF-ranked key phrases and
related-article suggestions (via `SemanticSearchEngine`).

### 2.3 Multilingual Intelligence (Module C)

**MultilingualProcessor** — `langdetect` for language identification, `deep-translator`
(Google Translate web backend) for translation-to-English. Every method degrades gracefully: if
the translation service is unreachable, `translate_to_english()` returns the original text with
a `translation_used: False` flag rather than raising — the same fail-safe pattern the midterm
used for its dataset download (public mirror with a built-in fallback sample).

### 2.4 Conversational Interface (Module D)

**ConversationalInterface** — a small set of regex-based intent rules (`count`, `summarize`,
`entity_lookup`, default `search`) extracts category/sentiment/entity-name/article-index
arguments from the query text and dispatches to the corresponding method on
`NewsBot2IntegratedSystem`. A rule-based router was chosen over a trained intent classifier
because the query vocabulary for this system is small and well defined (five intent shapes); the
same `ConversationalInterface.process()` entry point would work unchanged in front of a learned
classifier if the query surface grew.

### 2.5 Integration, Testing & Evaluation

**NewsBot2IntegratedSystem** exposes four methods: `.train()`, `.analyze_article()`,
`.search()`, `.ask()`. Training fits every learned component once and caches per-article
sentiment on `system.df` so downstream calls (filtering, conversation) don't recompute it.

**NewsBot2TestSuite** runs 10 sanity checks (one per component plus two end-to-end checks) and
reports pass/fail — a fast in-notebook signal that mirrors the pytest suite in `tests/`, which
adds 15 more granular unit tests (preprocessing, classifier held-out metrics, topic-modeling
output shapes, two full-system integration tests).

**NewsBot2Evaluator** computes: classifier held-out accuracy/macro-F1 plus 5-fold
cross-validation; a topic-distinctness proxy (average pairwise Jaccard overlap of each topic's
top-10 words — lower is more distinct); and average summarization compression ratio.

## 3. Installation Guide

```bash
git clone <your-repo-url>
cd ITAI2373-NewsBot-Final
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('vader_lexicon')"
pytest tests/ -v            # 15 tests, all passing
jupyter notebook notebooks/NewsBot_Intelligence_System_2.0.ipynb
```

Or open `notebooks/NewsBot_Intelligence_System_2.0.ipynb` directly in Google Colab — the first
cell installs everything needed.

## 4. Configuration Manual

All tunables live in `config/settings.py` / `NewsBot2Config`:

| Parameter | Default | Effect |
|---|---|---|
| `n_topics` | 8 | Number of LDA/NMF topics discovered |
| `topic_method` | `'lda'` | `'lda'` or `'nmf'` |
| `classifier_max_features` | 5000 | TF-IDF vocabulary size for the classifier |
| `semantic_dims` | 100 | LSA embedding dimensionality for semantic search |
| `entity_labels` | `(PERSON, ORG, GPE)` | spaCy entity types tracked in the relationship graph |
| `multi_label_threshold` | 0.20 | Minimum probability to include a category in multi-label output |
| `summarizer_ratio` | 0.3 | Target fraction of sentences kept in a summary |

## 5. Results (measured on this submission's run: 1,000 BBC News articles, 200/category)

| Metric | Value |
|---|---|
| Classifier held-out accuracy | 0.970 |
| Classifier held-out macro-F1 | 0.970 |
| 5-fold CV macro-F1 | ~0.96 (stable across folds) |
| Topics discovered | 8 (avg. pairwise word overlap 0.075 — distinct topics) |
| Entity graph | 4,074 entities (sampled on 400 articles) |
| Avg. summarization compression ratio | 0.294 (~3.4x shorter) |
| Internal test suite | 10/10 passed |
| Pytest suite | 15/15 passed |
| Training time (full pipeline) | ~24 seconds |

## 6. Known Limitations

- No publish dates in the BBC News dataset — every "evolution/trend" feature substitutes
  document order or category for time; the identical code applies once real timestamps exist.
- Semantic search uses LSA, not a transformer embedding model — a deliberate trade against an
  external model-hub dependency (see 2.2).
- Translation depends on a live external service (Google Translate via `deep-translator`) and
  has no offline model fallback beyond "show original text."
- The conversational interface's intent router is rule-based; it covers the query patterns
  demonstrated in the notebook but is not a general-purpose NLU system.
- Entity relationship mapping runs on a capped sample (400 of 1,000 articles) for runtime reasons
  on the Colab free tier; increasing `entity_sample_size` trades runtime for graph completeness.

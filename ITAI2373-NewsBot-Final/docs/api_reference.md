# API Reference

## src.newsbot2_system.NewsBot2Config
Configuration object. Key parameters: `n_topics`, `topic_method` (`'lda'`/`'nmf'`),
`classifier_max_features`, `semantic_dims`, `entity_labels`, `multi_label_threshold`,
`summarizer_ratio`.

## src.newsbot2_system.NewsBot2IntegratedSystem(config, spacy_model)
- `.train(df, text_col='content', category_col='category', entity_sample_size=300) -> dict`
  Fits every learned component; returns a training report (accuracy, F1, timing, etc.).
- `.analyze_article(text: str) -> dict`
  Full-pipeline analysis of a new article: language, classification, multi-label categories,
  sentiment, entities, summary, and an auto-generated brief.
- `.search(query: str, top_k=5) -> list[dict]`
  Semantic search over the trained corpus.
- `.filter_articles(category=None, sentiment=None) -> DataFrame`
  Filter the trained corpus by category and/or sentiment label.
- `.ask(query: str) -> dict`
  Natural-language query via the conversational interface.

## src.analysis.content_analysis
- `AdvancedNewsClassifier(max_features, multi_label_threshold)` — `.fit()`, `.evaluate()`,
  `.predict()`, `.predict_multi_label()`.
- `TopicDiscoveryEngine(n_topics, method)` — `.fit_transform()`, `.get_topic_words()`,
  `.get_all_topics()`, `.get_document_topics()`, `.topics_by_category()`.
- `SentimentEvolutionTracker(window)` — `.score()`, `.score_corpus()`, `.rolling_trend()`,
  `.evolution_by_category()`.
- `EntityRelationshipMapper(spacy_model)` — `.extract_entities()`, `.build_graph()`,
  `.top_entities()`, `.related_entities()`.

## src.language_models.language_understanding
- `IntelligentSummarizer(ratio, min_sentences, max_sentences)` — `.summarize(text) -> dict`.
- `SemanticSearchEngine(n_components)` — `.fit(documents)`, `.search(query, top_k)`,
  `.most_similar_to_doc(doc_index, top_k)`.
- `ContentEnhancer(semantic_search_engine)` — `.key_phrases(text, n)`, `.generate_brief(...)`.

## src.multilingual.multilingual_processor
- `MultilingualProcessor(target_language='en')` — `.detect_language(text)`,
  `.translate_to_english(text)`, `.analyze_multilingual_corpus(texts)`.

## src.conversation.conversational_interface
- `ConversationalInterface(integrated_system)` — `.classify_intent(query)`, `.process(query)`.

## src.data_processing
- `text_preprocessor.clean(text)`, `text_preprocessor.preprocess(text, ...)`
- `data_validator.validate_corpus(df, ...)`
- `feature_extractor.length_features(texts)`, `feature_extractor.punctuation_density(texts)`

## Testing / Evaluation
- `NewsBot2TestSuite(system).run_all() -> dict` — 10 internal sanity checks.
- `NewsBot2Evaluator(system).full_report() -> dict` — classifier, topic, and summarizer metrics.

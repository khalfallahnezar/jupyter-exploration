"""
NewsBot 2.0 - Module D: Conversational Interface
ConversationalInterface
"""
import re


class ConversationalInterface:
    """
    Rule-based intent classification + query processing for natural-language
    questions about the corpus, e.g. "show me positive tech news",
    "how many politics articles are there", "summarize article 42",
    "who is mentioned most with Google". A rule-based router is used instead
    of a trained intent classifier because the query vocabulary is small and
    well-defined here — the same interface can sit in front of a learned
    intent model without changing the rest of the system.
    """

    CATEGORY_WORDS = ['business', 'entertainment', 'politics', 'sport', 'sports', 'tech', 'technology']
    SENTIMENT_WORDS = {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}

    def __init__(self, integrated_system):
        self.system = integrated_system
        self.history = []

    def classify_intent(self, query):
        q = query.lower()
        if re.search(r'\bsummar', q):
            return 'summarize'
        if re.search(r'\bhow many\b|\bcount\b', q):
            return 'count'
        if re.search(r'\bwho is\b|\brelated to\b|\bconnected to\b|\bmentioned with\b', q):
            return 'entity_lookup'
        if re.search(r'\bfind\b|\bsearch\b|\babout\b|\bshow me\b', q):
            return 'search'
        return 'search'  # sensible default: treat unrecognized queries as a search

    def _extract_category(self, q):
        for word in self.CATEGORY_WORDS:
            if word in q:
                return 'sport' if word == 'sports' else ('tech' if word == 'technology' else word)
        return None

    def _extract_sentiment(self, q):
        for word, label in self.SENTIMENT_WORDS.items():
            if word in q:
                return label
        return None

    def _extract_number(self, q):
        m = re.search(r'\d+', q)
        return int(m.group()) if m else None

    def process(self, query):
        self.history.append(query)
        intent = self.classify_intent(query)
        q = query.lower()
        category = self._extract_category(q)
        sentiment = self._extract_sentiment(q)

        if intent == 'count':
            df = self.system.filter_articles(category=category, sentiment=sentiment)
            response = f"Found {len(df)} article(s)"
            response += f" in {category}" if category else ""
            response += f" with {sentiment.lower()} sentiment" if sentiment else ""
            response += "."
            return {'intent': intent, 'response': response, 'result_count': len(df)}

        if intent == 'summarize':
            idx = self._extract_number(q)
            if idx is None or idx >= len(self.system.df):
                return {'intent': intent, 'response': "Tell me which article number to summarize, e.g. 'summarize article 12'."}
            text = self.system.df.iloc[idx]['content']
            summary = self.system.summarizer.summarize(text)['summary']
            return {'intent': intent, 'response': summary, 'doc_index': idx}

        if intent == 'entity_lookup':
            words = re.findall(r'[A-Z][a-zA-Z]+', query)
            entity = words[0] if words else None
            if entity is None:
                return {'intent': intent, 'response': "Which person, organization, or place are you asking about?"}
            related = self.system.entity_mapper.related_entities(entity, n=5)
            if not related:
                return {'intent': intent, 'response': f"No strong co-occurrence relationships found for '{entity}'."}
            names = ', '.join(f"{r['entity']} ({r['co_occurrences']}x)" for r in related)
            return {'intent': intent, 'response': f"'{entity}' most often appears alongside: {names}.", 'entity': entity}

        # default: search
        results = self.system.search(query, top_k=5)
        if category or sentiment:
            df = self.system.filter_articles(category=category, sentiment=sentiment)
            preview = df['content'].head(5).apply(lambda t: t[:120] + '...').tolist()
            return {'intent': 'search', 'response': f"Found {len(df)} matching article(s).",
                     'category': category, 'sentiment': sentiment, 'previews': preview}
        response = f"Top matches for '{query}':\n" + "\n".join(
            f"- (doc {r['doc_index']}, similarity {r['similarity']}) {r['preview']}" for r in results)
        return {'intent': intent, 'response': response, 'results': results}

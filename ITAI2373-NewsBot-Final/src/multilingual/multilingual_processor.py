"""
NewsBot 2.0 - Module C: Multilingual Intelligence
MultilingualProcessor
"""
from langdetect import detect, detect_langs, LangDetectException

try:
    from deep_translator import GoogleTranslator
    _TRANSLATION_AVAILABLE = True
except Exception:
    _TRANSLATION_AVAILABLE = False

_LANGUAGE_NAMES = {
    'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German', 'ar': 'Arabic',
    'zh-cn': 'Chinese', 'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
    'hi': 'Hindi', 'nl': 'Dutch', 'tr': 'Turkish', 'ko': 'Korean',
}


class MultilingualProcessor:
    """
    Advanced multilingual processing: automatic language detection plus
    translation-to-English so the rest of the NewsBot pipeline (which is
    trained on English text) can analyze non-English sources. Translation
    calls an external service, so every method degrades gracefully — matching
    the same "always runs end-to-end, even offline" pattern used for the BBC
    dataset loader in the midterm project.
    """

    def __init__(self, target_language='en'):
        self.target_language = target_language

    def detect_language(self, text):
        try:
            code = detect(text)
            candidates = detect_langs(text)
            confidence = round(float(candidates[0].prob), 3) if candidates else None
            return {'language_code': code, 'language_name': _LANGUAGE_NAMES.get(code, code),
                    'confidence': confidence}
        except LangDetectException:
            return {'language_code': 'unknown', 'language_name': 'Unknown', 'confidence': 0.0}

    def translate_to_english(self, text):
        detected = self.detect_language(text)
        if detected['language_code'] == self.target_language:
            return {'translated_text': text, 'source_language': detected['language_name'],
                     'translation_used': False}
        if not _TRANSLATION_AVAILABLE:
            return {'translated_text': text, 'source_language': detected['language_name'],
                     'translation_used': False, 'note': 'translator unavailable — showing original text'}
        try:
            translated = GoogleTranslator(source='auto', target=self.target_language).translate(text)
            return {'translated_text': translated, 'source_language': detected['language_name'],
                     'translation_used': True}
        except Exception as e:
            return {'translated_text': text, 'source_language': detected['language_name'],
                     'translation_used': False, 'note': f'translation failed ({e}) — showing original text'}

    def analyze_multilingual_corpus(self, texts):
        """Language breakdown across a batch of documents — supports the
        cross-language coverage comparison use case."""
        from collections import Counter
        langs = [self.detect_language(t)['language_name'] for t in texts]
        return dict(Counter(langs))

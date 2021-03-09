import re
from langdetect import detect

from features.FeatureFunction import FeatureFunction

class LanguageCount(FeatureFunction):

    def __init__(self):
        super().__init__()

    def evaluate(self, context: str) -> float:
        sentences = re.split("\.|!|\?", context)
        languages = set()
        for sentence in sentences:
            sentence = sentence.strip()
            words = sentence.split()
            for word in words:
                try:
                    languages.add(detect(word))
                except:
                    pass
        
        return len(languages)

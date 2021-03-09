import re

from features.FeatureFunction import FeatureFunction

class WordVariety(FeatureFunction):

    def __init__(self):
        super().__init__()

    def evaluate(self, context: str) -> float:
        sentences = re.split("\.|!|\?", context)
        num_words = 0
        unique_words = set()
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            words = sentence.split()
            for word in words:
                if word.isalpha():
                    unique_words.add(word.lower())
            num_words += len(words)
        return len(unique_words)/num_words

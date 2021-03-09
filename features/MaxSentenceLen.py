import re

from features.FeatureFunction import FeatureFunction

class MaxSentenceLen(FeatureFunction):

    def __init__(self):
        super().__init__()

    def evaluate(self, context: str) -> float:
        sentences = re.split("\.|!|\?", context)
        max_sentence = max(sentences, key=len)
        return len(max_sentence.split())

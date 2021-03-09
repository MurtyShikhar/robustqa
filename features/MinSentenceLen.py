import re

from features.FeatureFunction import FeatureFunction

class MinSentenceLen(FeatureFunction):

    def __init__(self):
        super().__init__()

    def evaluate(self, context: str) -> float:
        sentences = re.split("\.|!|\?", context)
        min_sentence = list(min(sentences, key=len))
        return len(max_sentence)

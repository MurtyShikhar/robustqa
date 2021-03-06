import re

from clustering.FeatureFunction import FeatureFunction


class AvgSentenceLen(FeatureFunction):

    def __init__(self):
        super().__init__()

    def evaluate(self, context: str) -> float:
        sentences = re.split("\.|!|\?", context)
        num_words = 0
        num_sentences = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            num_words += len(sentence.split())
            num_sentences += 1
        return num_words/num_sentences

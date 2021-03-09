import re

from features.FeatureFunction import FeatureFunction


class NumberOfCommas(FeatureFunction):

    def __init__(self):
        super().__init__()

    def evaluate(self, context: str) -> float:
        sentences = re.split("\.|!|\?", context)
        num_commas = 0
        num_words = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            words = sentence.split()
            for word in words:
                if len(word) > 0 and word[-1] == ",":
                    num_commas += 1
            num_words += len(sentence)
        return num_commas/num_words

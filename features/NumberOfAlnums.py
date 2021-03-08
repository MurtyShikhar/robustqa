import re

from features.FeatureFunction import FeatureFunction


class NumberOfAlnums(FeatureFunction):

    def __init__(self):
        super().__init__()

    def evaluate(self, context: str) -> float:
        sentences = re.split("\.|!|\?", context)
        num_alnums = 0
        num_words = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            words = sentence.split()
            for word in words:
                if len(word) > 0 and word.isalnum() and not word.isalpha():
                    num_alnums += 1
            num_words += len(sentence)
        return num_alnums/num_words

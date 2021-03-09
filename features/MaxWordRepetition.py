import re

from features.FeatureFunction import FeatureFunction

class MaxWordRepetition(FeatureFunction):

    def __init__(self):
        super().__init__()

    def evaluate(self, context: str) -> float:
        sentences = re.split("\.|!|\?", context)
        max_word_repetition = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            words = sentence.split()
            prev_word = "3"  # alpha words will never be equal to 3
            cur_word_repetition = 1
            for word in words:
                if word.isalpha() and word == prev_word:
                    cur_word_repetition += 1
                else:
                    max_word_repetition = max(max_word_repetition, cur_word_repetition)
                    prev_word = word
                    cur_word_repetition = 1
            max_word_repetition = max(max_word_repetition, cur_word_repetition)
        return max_word_repetition

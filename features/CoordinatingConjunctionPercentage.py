import re
import nltk

from nltk_tagger import count_tags
from FeatureFunction import FeatureFunction

class CoordinatingConjunctionPercentage(FeatureFunction):

    def __init__(self):
        nltk.download('averaged_perceptron_tagger')
        super().__init__()

    def evaluate(self, context: str) -> float:
        return count_tags(context, ['CC']) / len(context.split())

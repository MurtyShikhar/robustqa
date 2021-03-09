import re
from nltk_tagger import count_tags

from features.FeatureFunction import FeatureFunction

class CoordinatingConjunctionPercentage(FeatureFunction):

    def __init__(self):
        super().__init__()

    def evaluate(self, context: str) -> float:
        return count_tags(context, 'CC') / len(context.split())

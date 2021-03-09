import re
import nltk

from features.nltk_tagger import count_tags
from features.FeatureFunction import FeatureFunction

class NounPercentage(FeatureFunction):

    def __init__(self):
        nltk.download('averaged_perceptron_tagger')
        super().__init__()

    def evaluate(self, context: str) -> float:
        return count_tags(context, ['NN', 'NNS', 'NNP', 'NNPS']) / len(context.split())

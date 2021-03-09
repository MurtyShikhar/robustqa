import re
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

from FeatureFunction import FeatureFunction

class SentimentAnalysis(FeatureFunction):

    def __init__(self):
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        super().__init__()

    def evaluate(self, context: str) -> float:
        sentences = re.split("\.|!|\?", context)
        scores = [
            self.sia.polarity_scores(sentence)["compound"]
            for sentence in sentences
        ]

        return np.mean(scores)

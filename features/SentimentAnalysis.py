import re
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

from features.FeatureFunction import FeatureFunction

class SentimentAnalysis(FeatureFunction):

    def __init__(self):
        super().__init__()

    def evaluate(self, context: str) -> float:
        nltk.download('vader_lexicon')
        sentences = re.split("\.|!|\?", context)
        scores = [
            sia.polarity_scores(sentence)["compound"]
            for sentence in sentences
        ]

        return np.mean(scores)

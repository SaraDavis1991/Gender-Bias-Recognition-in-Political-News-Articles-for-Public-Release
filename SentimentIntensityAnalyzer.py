import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalyzer(): 

    def __init__(self): 
        self.Analyzer = SentimentIntensityAnalyzer()

    def AnalyzeSentiment(self, article: str):
        intensity = self.Analyzer.polarity_scores(article)
        return intensity

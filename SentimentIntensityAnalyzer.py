import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#import flair 

class SentimentAnalyzer(): 

    def __init__(self): 
        self.Analyzer = SentimentIntensityAnalyzer()
        self.Flair = flair.models.TextClassifier.load('en-sentiment')

    def AnalyzeSentiment(self, article: str):
        intensity = self.Analyzer.polarity_scores(article)
        return intensity

    def AnalyzeSentiment2(self, article: str):
 #       sentence = flair.data.Sentence(article) 
  #      self.Flair.predict(sentence)
    	pass     
   #     return sentence.labels

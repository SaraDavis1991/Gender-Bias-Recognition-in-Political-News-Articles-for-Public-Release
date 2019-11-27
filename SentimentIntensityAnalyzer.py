import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize 
import statistics

class SentimentAnalyzer(): 

    def __init__(self): 
        self.Analyzer = SentimentIntensityAnalyzer()

    def AnalyzeSentiment(self, article: str, should_average_intensities=True):

        polarities = [] 
        tokens = tokenize.sent_tokenize(article) 
        
        neg_list = []
        pos_list = []
        neu_list = []

        for sent in tokens: 

            intensity = self.Analyzer.polarity_scores(sent)
            polarities.append(intensity) 

            neg_list.append(intensity['neg'])
            pos_list.append(intensity['pos'])
            neu_list.append(intensity['neu'])
        
        if should_average_intensities:

            results = {}
            
            results['neg_mean'] = statistics.mean(neg_list)
            results['pos_mean'] = statistics.mean(pos_list)
            results['neu_mean'] = statistics.mean(neu_list)
            results['neg_median'] = statistics.median(neg_list)
            results['pos_median'] = statistics.median(pos_list)
            results['neu_median'] = statistics.median(neu_list)

            return polarities, results    

        return polarities
    
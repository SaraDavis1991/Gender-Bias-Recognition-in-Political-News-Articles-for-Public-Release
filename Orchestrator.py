#classes

from DataReader import DataReader
from DataContracts import Article
from doc2vec import doc
from SentimentIntensityAnalyzer import SentimentAnalyzer
from Metrics import Metrics
from Visualizer import Visualizer 
import ApplicationConstants

#models
from Models.SVM_engine import SVM
from Models.KNN_engine import KNN
from Models.Naive_Bayes_engine import Naive_Bayes
from Models.Linear_Classification_engine import Linear_Classifier 
from Models.NN_engine import NN

#helpers
import statistics
import numpy as np 
import matplotlib.pyplot as plt 

class Orchestrator():

    def __init__(self):
        self.Reader = DataReader()
        
        self.Splits = None 
        self.Sources = None
        self.docEmbed = doc()
        self.Metrics = Metrics()
        self.Visualizer = Visualizer() 
        self.SentimentAnalyzer = SentimentAnalyzer() 
        
    def read_data(self, clean=True, number_of_articles = 50):       
        return self.Reader.Load_Splits(ApplicationConstants.all_articles_random, clean=clean, number_of_articles=number_of_articles)
    
    def embed_fold(self, articles, labels):
        ''' 
        trains and returns the vector embeddings for doc2vec or sent2vec 

        Parameters:5
        articles: a list of articles that are cleaned
        labels: a list of labels corresponding to the article genders
        ''' 

        #emb = self.docEmbed.word2vec() 
        targets, regressors = self.docEmbed.Embed(articles, labels)

        return list(targets), regressors
    
    def calc_sent(self, sentiment):
        if sentiment["pos_mean"] > sentiment["neg_mean"]:
            return 0
        else:
            return 1
        #return max(sentiment["pos_mean"], sentiment["neg_mean"])
        
    def graph_sentiment(self, leaning, Fsentiment, Msentiment):

        femaleVals = []
        maleVals = []

        for i in range(len(Fsentiment)):
            femaleVals.append(self.calc_sent(Fsentiment[i]))
        for j in range(len(Msentiment)):
            maleVals.append(self.calc_sent(Msentiment[j]))

        female_pos = list(filter(lambda sent: sent == 0, femaleVals))
        female_neg = list(filter(lambda sent: sent == 1, femaleVals))
        female_bars = [len(female_pos), len(female_neg)]
        male_pos = list(filter(lambda sent: sent == 0, maleVals))
        male_neg = list(filter(lambda sent: sent == 1, maleVals))
        male_bars = [len(male_pos), len(male_neg)]

        N = 2
        ind = np.arange(N) 
        width = 0.35 

        plt.bar(ind[0], len(female_pos), width, align='center', color='green')
        plt.bar(ind[0] + width, len(female_neg), width, align='center', color='blue')
        plt.bar(ind[1], len(male_pos), width, align='center', label='positive sentiment', color='green')
        plt.bar(ind[1] + width, len(male_neg), width, align='center', label='negative sentiment', color='blue')

        plt.ylabel('Article Polarity Count')
        plt.title('Counts of positve and negative sentiment per gender for ' + leaning)
        plt.xticks(ind + width / 2, ('Female Sentiment Polarity', 'Male Sentiment Polarity'))
        plt.legend(loc='best')
        plt.show()

    def Retrieve_mean_sentiment(self, sentiments):

        results = {}

        neu_mean = []
        pos_mean = []
        neg_mean = [] 
        neg_median = []
        pos_median = []
        neu_median = []

        for sentiment in sentiments:

            neu_mean.append(sentiment['neu_mean'])
            neg_mean.append(sentiment['neg_mean'])
            pos_mean.append(sentiment['pos_mean'])

            neu_median.append(sentiment['neu_median'])
            neg_median.append(sentiment['neg_median'])
            pos_median.append(sentiment['pos_median'])   

        results['neg_mean'] = statistics.mean(neg_mean)
        results['pos_mean'] = statistics.mean(pos_mean)
        results['neu_mean'] = statistics.mean(neu_mean)

        results['neg_median'] = statistics.median(neg_median)
        results['pos_median'] = statistics.median(pos_median)
        results['neu_median'] = statistics.median(neu_median)

        return results

    def run_sentiment_analysis_all(self, articles):
     
        #separate per sources per gender combining datasets
        results = {}

        for leaning in articles: 
            
            results[leaning] = {} 
            all_articles_for_leaning = articles[leaning][ApplicationConstants.Test] + articles[leaning][ApplicationConstants.Validation] + articles[leaning][ApplicationConstants.Train]

            #separte per gender
            female_articles = list(filter(lambda article: article.Label.TargetGender == 0, all_articles_for_leaning))
            male_articles = list(filter(lambda article: article.Label.TargetGender == 1, all_articles_for_leaning))
            female_sentiments = []
            male_sentiments = []

            for article in female_articles:
                
                if (article.Content != '' and not article.Content.isspace()):
                    _, sentiment = self.SentimentAnalyzer.AnalyzeSentiment(article.Content)
                    print(sentiment)
                    female_sentiments.append(sentiment)
            
            for article in male_articles:
                if (article.Content != '' and not article.Content.isspace()):
                    _, sentiment = self.SentimentAnalyzer.AnalyzeSentiment(article.Content)
                    male_sentiments.append(sentiment)
            
            self.graph_sentiment(leaning, female_sentiments, male_sentiments)
            female_sentiment_scores = self.Retrieve_mean_sentiment(female_sentiments)
            male_sentiment_scores = self.Retrieve_mean_sentiment(male_sentiments)

            print ('female:',female_sentiment_scores)
            print ('male:', male_sentiment_scores)

    def train_all(self, splits):
        ''' trains all models against all leanings
        
        Parameters: 
        ------------
        splits: A list of the splits 

        ''' 
        models = [SVM(), KNN(), Naive_Bayes(), Linear_Classifier(), NN()]

        #models = [NN()]

        split_count = 0 

        #for each split
        for split in splits:
            
            print("Starting split:", str(split_count), "\n")
            split_count += 1

            #loop over all leanings
            for leaning in split:

                print("For leaning:", leaning.upper())
                
                #train embeddings
                training_dataset = split[leaning][ApplicationConstants.Train]
                training_labels, training_embeddings = self.embed_fold(list(map(lambda article: article.Content, training_dataset)), list(map(lambda article: article.Label.TargetGender, training_dataset)))

                #validation embeddings 
                validation_dataset = split[leaning][ApplicationConstants.Validation]
                validation_labels, validation_embeddings = self.embed_fold(list(map(lambda article: article.Content, validation_dataset)), list(map(lambda article: article.Label.TargetGender, validation_dataset)))

                #test embeddings
                test_dataset = split[leaning][ApplicationConstants.Test]
                test_labels, test_embeddings = self.embed_fold(list(map(lambda article: article.Content, test_dataset)), list(map(lambda article: article.Label.TargetGender, test_dataset)))

                for model in models: 

                    #get prediction from embeddings 
                    model.Train(training_embeddings, training_labels, validation_embeddings, validation_labels)
                    prediction = model.Predict(test_embeddings)
                    print("Model:", str(type(model)).split('.')[2].split('\'')[0], "Accuracy:", self.Metrics.Accuracy(prediction, test_labels), "F-Measure:", self.Metrics.Fmeasure(prediction, test_labels))   

                #model = models[0] 
                #model.Model.coefs_[model.Model.n_layers_ - 2]
                #self.Visualizer.plot_TSNE(training_embeddings, training_labels)

orchestrator = Orchestrator()

splits = orchestrator.read_data(clean=False, number_of_articles=10) 

orchestrator.run_sentiment_analysis_all(splits[0]) 
orchestrator.train_all(splits)



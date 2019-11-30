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
import os.path

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
        score = sentiment[0]
        magnitude = sentiment[1]

        if score > 0.1:
            return 'pos'
        elif score < -0.1:
            return 'neg'
        
    def graph_sentiment(self, Fsentiment, Msentiment):

        pos_counts_per_leaning_female = [] 
        neg_counts_per_leaning_male = []
        pos_counts_per_leaning_male = [] 
        neg_counts_per_leaning_female = []
        leanings = ["Breitbart", "Fox", "USA Today", "New York Times", "Huffpost"]

        breitbart_female_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.Breitbart, Fsentiment))))
        breitbart_male_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.Breitbart, Msentiment))))
        fox_female_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.Fox, Fsentiment))))
        fox_male_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.Fox, Msentiment))))
        usa_female_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.usa_today, Fsentiment))))
        usa_male_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.usa_today, Msentiment))))
        nyt_female_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.New_york_times, Fsentiment))))
        nyt_male_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.New_york_times, Msentiment))))
        hp_female_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.HuffPost, Fsentiment))))
        hp_male_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.HuffPost, Msentiment))))

        male_leanings = [breitbart_male_sentiments, fox_male_sentiments, usa_male_sentiments, nyt_male_sentiments, hp_male_sentiments]
        female_leanings = [breitbart_female_sentiments, fox_female_sentiments, usa_female_sentiments, nyt_female_sentiments, hp_female_sentiments]

        for leaning in range(5): 

            femaleVals = []
            maleVals = []

            for sentiment in female_leanings[leaning][0]:
                femaleVals.append(self.calc_sent(sentiment))

            for sentiment in male_leanings[leaning][0]:
                maleVals.append(self.calc_sent(sentiment))

            female_pos = len(list(filter(lambda sent: sent == 'pos', femaleVals)))
            female_neg = len(list(filter(lambda sent: sent == 'neg', femaleVals)))
            male_pos = len(list(filter(lambda sent: sent == 'pos', maleVals)))
            male_neg = len(list(filter(lambda sent: sent == 'neg', maleVals)))

            pos_counts_per_leaning_female.append(female_pos / 125)
            neg_counts_per_leaning_female.append(female_neg / 125)
            neg_counts_per_leaning_male.append(male_neg / 125)
            pos_counts_per_leaning_male.append(male_pos / 125)

        plt.plot(leanings, pos_counts_per_leaning_female, marker='D', label='Positive Female Articles', color='seagreen')
        plt.plot(leanings, neg_counts_per_leaning_female, marker='D', label='Negative Female Articles', color='slateblue')
        plt.plot(leanings, pos_counts_per_leaning_male, marker='D', label='Positive Male Articles', color='orange')
        plt.plot(leanings, neg_counts_per_leaning_male, marker='D', label='Negative Male Articles', color='crimson')

        plt.ylabel('Mean Leaning Sentiment Positive:Negative Ratio')
        plt.title('Positive and Negative Sentiment by Leaning and Gender')
        plt.xticks(leanings)
        plt.ylim((0, 1))
        plt.legend(loc='center right')
        plt.show()

    def run_sentiment_analysis_all(self, articles):
     
        #separate per sources per gender combining datasets
        results = {}
        all_female = []
        all_male = []

        for leaning in articles: 
            
            results[leaning] = {} 
            all_articles_for_leaning = articles[leaning][ApplicationConstants.Test] + articles[leaning][ApplicationConstants.Validation] + articles[leaning][ApplicationConstants.Train]

            #separte per gender
            female_articles = list(filter(lambda article: article.Label.TargetGender == 0, all_articles_for_leaning))
            male_articles = list(filter(lambda article: article.Label.TargetGender == 1, all_articles_for_leaning))

            female_sentiments = []
            male_sentiments = []
            print(leaning)

            female_path = './sentiment/' + leaning + '_female_sentiment_cleaned'
            male_path = './sentiment/' + leaning + '_male_sentiment_cleaned'

            if (not os.path.isfile(female_path + '.npy')):
                for article in female_articles:
                    
                    if (article.Content != '' and not article.Content.isspace()):
                        score, magnitude = self.SentimentAnalyzer.AnalyzeSentiment(article.Content)
                        female_sentiments.append((score, magnitude)) 
                
                np.save(female_path, female_sentiments)
            else:
                female_sentiments = np.load(female_path + '.npy')
                female_sentiments = list(map(lambda article: (article[0], article[1]), female_sentiments))
             
            if (not os.path.isfile(male_path + '.npy')):
                for article in male_articles:
                    if (article.Content != '' and not article.Content.isspace()):
                        score, magnitude = self.SentimentAnalyzer.AnalyzeSentiment(article.Content)
                        male_sentiments.append((score, magnitude))
                    
                np.save(male_path, male_sentiments)

            else:
                male_sentiments = np.load(male_path + '.npy')
                male_sentiments = list(map(lambda article: (article[0], article[1]), male_sentiments))

            all_female.append((leaning, female_sentiments))
            all_male.append((leaning, male_sentiments))

        self.graph_sentiment(all_female, all_male)

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
                self.Visualizer.plot_TSNE(leaning, training_embeddings + validation_embeddings + test_embeddings, training_labels + validation_labels + test_labels)

orchestrator = Orchestrator()
splits = orchestrator.read_data(clean=True, number_of_articles=25) 
#orchestrator.run_sentiment_analysis_all(splits[0]) 
orchestrator.train_all(splits)



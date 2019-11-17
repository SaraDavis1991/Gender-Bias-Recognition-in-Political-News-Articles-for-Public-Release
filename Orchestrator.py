from DataReader import DataReader

from preprocessor import Preprocessor
from DataContracts import Article
from doc2vec import doc

#models
from Models.SVM_engine import SVM
from Models.KNN_engine import KNN
from Models.Naive_Bayes_engine import Naive_Bayes
from Models.Linear_Regression_engine import Linear_Regression 
from Models.NN_engine import NN

import ApplicationConstants

class Orchestrator():

    def __init__(self):
        self.Reader = DataReader()
        self.Preprocessor = Preprocessor()
        self.Splits = None 
        self.Sources = None 
        self.docEmbed = doc()
        
    def read_data(self):       
        return self.Reader.Load_Splits(ApplicationConstants.all_articles)

    def clean_all(self, splits):
        ''' cleans all data within the splits per leaning '''

        #for each leaning
        for leaning in splits: 

            #get train, test, val 
            for dataset in splits[leaning]:

                #get article content
                articles = splits[leaning][dataset]
                
                #clean, putting cleaned data back into the split dictionary
                for index, article in enumerate(articles):

                    content = article.Content
                    cleaned_content = orchestrator.Preprocessor.Clean(content)
                    splits[leaning][dataset][index].Content = cleaned_content
    
    def embed_fold(self, articles, labels):
        ''' 
        trains and returns the vector embeddings for doc2vec or sent2vec 
        param: articles: a list of articles that are cleaned
        param: labels: a list of labels corresponding to the article genders
        ''' 

        targets, regressors = self.docEmbed.Embed(articles, labels)

        return targets, regressors

    def train_all(self, split_data):
        ''' trains all models against all leanings ''' 

        models = [SVM(), KNN(), Naive_Bayes(), Linear_Regression(), NN()]

        for leaning in split_data:

            #train embeddings
            training_dataset = split_data[leaning][ApplicationConstants.Train]
            training_labels, training_embeddings = self.embed_fold(list(map(lambda article: article.Content, training_dataset)), list(map(lambda article: article.Label.TargetGender, training_dataset)))

            #validation embeddings 
            validation_dataset = split_data[leaning][ApplicationConstants.Validation]
            validation_labels = list(map(lambda article: article.Label.TargetGender, validation_dataset))
            #_, validation_embeddings = self.embed_fold(list(map(lambda article: article.Content, validation_dataset)), validation_labels)

            #test embeddings
            test_dataset = split_data[leaning][ApplicationConstants.Test]
            test_labels, test_embeddings = self.embed_fold(list(map(lambda article: article.Content, test_dataset)), list(map(lambda article: article.Label.TargetGender, test_dataset)))

            for model in models: 

                model.Train(training_embeddings, training_labels)
                prediction = model.Predict(test_embeddings)

                print(model.Accuracy(prediction, test_labels))               

orchestrator = Orchestrator()
splits = orchestrator.read_data() 
#orchestrator.clean_all(splits)

orchestrator.train_all(splits)


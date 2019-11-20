import sys

from DataReader import DataReader

from preprocessor import Preprocessor
from DataContracts import Article
from doc2vec import doc

#models
from Models.SVM_engine import SVM
from Models.KNN_engine import KNN
from Models.Naive_Bayes_engine import Naive_Bayes
from Models.Linear_Classification_engine import Linear_Classifier 
from Models.NN_engine import NN

#metrics
from Metrics import Metrics

#visualizations
from Visualizer import Visualizer 

import ApplicationConstants

class Orchestrator():

    def __init__(self):
        self.Reader = DataReader()
        self.Preprocessor = Preprocessor()
        self.Splits = None 
        self.Sources = None
        self.docEmbed = doc()
        self.Metrics = Metrics()
        self.Visualizer = Visualizer() 
        
    def read_data(self):       
        return self.Reader.Load_Splits(ApplicationConstants.all_articles)

    def clean_all(self, splits):
        ''' cleans all data within the splits per leaning '''

        print("Cleaning ", end='')
        sys.stdout.flush()

        for index, split in enumerate(splits): 

            #for each leaning
            for leaning in split: 

                #get train, test, val 
                for dataset in split[leaning]:

                    print(' . ', end='')
                    sys.stdout.flush()

                    #get article content
                    articles = split[leaning][dataset]
                    
                    #clean, putting cleaned data back into the split dictionary
                    for index, article in enumerate(articles):
                        
                        #convert labels to ints 
                        if (article.Label.TargetGender == ApplicationConstants.Female):
                            article.Label.TargetGender = 0
                        elif (article.Label.TargetGender == ApplicationConstants.Male):
                            article.Label.TargetGender = 1
 
                        content = article.Content
                        cleaned_content = orchestrator.Preprocessor.Clean(content)
                        split[leaning][dataset][index].Content = cleaned_content

        print("\nDone!\n")
    
    def embed_fold(self, articles, labels):
        ''' 
        trains and returns the vector embeddings for doc2vec or sent2vec 

        Parameters:
        articles: a list of articles that are cleaned
        labels: a list of labels corresponding to the article genders
        ''' 

        targets, regressors = self.docEmbed.Embed(articles, labels)

        return list(targets), regressors

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

                    model.Train(training_embeddings, training_labels, validation_embeddings, validation_labels)
                    prediction = model.Predict(test_embeddings)
                    print("Model:", str(type(model)).split('.')[2].split('\'')[0], "Accuracy:", self.Metrics.Accuracy(prediction, test_labels), "F-Measure:", self.Metrics.Fmeasure(prediction, test_labels))   

                print('\n')
                #model = models[0] 
                #model.Model.coefs_[model.Model.n_layers_ - 2]
                #self.Visualizer.plot_TSNE(training_embeddings, training_labels)

orchestrator = Orchestrator()
splits = orchestrator.read_data() 
orchestrator.clean_all(splits)

orchestrator.train_all(splits)


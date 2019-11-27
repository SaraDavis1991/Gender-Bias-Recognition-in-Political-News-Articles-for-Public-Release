#classes
from DataReader import DataReader

from DataContracts import Article
from doc2vec import doc
from SentimentIntensityAnalyzer import SentimentAnalyzer

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

#Constants
import ApplicationConstants

class Orchestrator():

    def __init__(self):
        self.Reader = DataReader()
        
        self.Splits = None 
        self.Sources = None
        self.docEmbed = doc()
        self.Metrics = Metrics()
        self.Visualizer = Visualizer() 
        self.SentimentAnalyzer = SentimentAnalyzer() 
        
    def read_data(self, clean=True):       
        return self.Reader.Load_Splits(ApplicationConstants.all_articles, clean=clean)
    
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

    def run_sentiment_analysis(self, articles):
     
        positive_sum = 0 
        negative_sum = 0 

        for article in articles:
            
            result = self.SentimentAnalyzer.AnalyzeSentiment(article.Content)
            
            prediction = result[0].value

            if (prediction == "POSITIVE"):
                positive_sum += 1
            else:
                negative_sum += 1

        print ("Pos:", positive_sum / len(articles), "; Neg:", negative_sum / len(articles))

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
                #get sentiment 
                #male_articles = list(filter(lambda article: article.Label.TargetGender == 1, test_dataset + training_dataset + validation_dataset))
                #female_articles = list(filter(lambda article: article.Label.TargetGender == 0, test_dataset + training_dataset + validation_dataset))

                #self.run_sentiment_analysis(male_articles)
                #self.run_sentiment_analysis(female_articles)

orchestrator = Orchestrator()
splits = orchestrator.read_data(clean=False) 
sentimentAnalyzer = SentimentAnalyzer() 
barrack_articles = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.SarahPalin, splits[0]['fox']['test']))[:5]
don_articles = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.HillaryClinton, splits[0]['fox']['train']))[:5]

positive_sum = 0 
negative_sum = 0 
neu_sum = 0
for article in don_articles:
    
    result = sentimentAnalyzer.AnalyzeSentiment2(article.Content)
    
    prediction = result[0].value

    # positive_sum += result['pos']
    # negative_sum += result['neg']
    # neu_sum += result['neu']

    if (prediction == "POSITIVE"):
        positive_sum += 1
    else:
        negative_sum += 1

print ("Pos:", positive_sum / len(don_articles), "; Neg:", negative_sum / len(don_articles))

positive_sum = 0 
negative_sum = 0 
neu_sum = 0
for article in barrack_articles:
    
    result = sentimentAnalyzer.AnalyzeSentiment2(article.Content)
    
    # positive_sum += result['pos']
    # negative_sum += result['neg']
    # neu_sum += result['neu']

    prediction = result[0].value

    if (prediction == "POSITIVE"):
        positive_sum += 1
    else:
        negative_sum += 1

print ("Pos:", positive_sum / len(barrack_articles), "; Neg:", negative_sum / len(barrack_articles))
orchestrator.train_all(splits)


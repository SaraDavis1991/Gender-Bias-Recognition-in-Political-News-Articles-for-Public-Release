#classes

from DataReader import DataReader
from DataContracts import Article
from doc2vec import doc
from SentimentIntensityAnalyzer import SentimentAnalyzer
from Metrics import Metrics
from Visualizer import Visualizer 
from imdb_data import LabeledLineSentence
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

    def imdb(self):
        sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS' }
        sentences = LabeledLineSentence(sources)
        vectors, labels = sentences.generate_imdb_vec()
        return vectors, labels

    def train_sent_models(self, imdb_vec, labels ):
         models = [ Linear_Classifier(), NN()]
         #print(imdb_vec)
         for model in models:
            model.Train(imdb_vec, labels, None, None)
            allF = []
            allM = []

            for split in splits[:1]:
            
                    #print("Starting split:", str(split_count), "\n")
                    #split_count += 1

                    #loop over all leanings
                    for leaning in split:
                        male = []
                       
                        female = []
                        
                        print("For leaning:", leaning.upper())
                        
                        #train embeddings
                        training_dataset = split[leaning][ApplicationConstants.Train]
                     
                        validation_dataset = split[leaning][ApplicationConstants.Validation]
                       
                        test_dataset = split[leaning][ApplicationConstants.Test]
                        
              

                        fucking_labels, fucking_embeddings, fucking_model = self.embed_fold(list(map(lambda article: article.Content, training_dataset + validation_dataset + test_dataset)), list(map(lambda article: article.Label.TargetGender, training_dataset + validation_dataset + test_dataset)), 0, leaning)
                        
                        predictions = model.Predict(fucking_embeddings)

                        for i in range(len(predictions)):
                            if fucking_labels[i] == 0:
                                female.append(predictions[i])
                            elif fucking_labels[i] == 1:
                                male.append(predictions[i])
                            

                        allF.append((leaning, female))
                        allM.append((leaning, male))
            self.graph_sentiment(allF, allM)





    def embed_fold(self, articles, labels, fold, leaning):
        ''' 
        trains and returns the vector embeddings for doc2vec or sent2vec 

        Parameters:5
        articles: a list of articles that are cleaned
        labels: a list of labels corresponding to the article genders
        ''' 

        #emb = self.docEmbed.word2vec() 
        targets, regressors, model = self.docEmbed.Embed(articles, labels, fold, leaning)

        return list(targets), regressors, model
    
    def calc_sent(self, sentiment):

        if sentiment == 0:
            return 'neg'
        else:
            return 'pos'
        #score = sentiment[0]
        #magnitude = sentiment[1]

        #if score > 0.25:
        #    return 'pos'
        #elif score < -0.25:
        #    return 'neg'
        
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
            print(leaning)
            print("Num Female Pos: " + str(female_pos))
            print("Num Female Neg: " + str(female_neg))
            print("Num Male Pos: " + str(male_pos))
            print("Num Male Neg: " + str(male_neg))


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

    def embed_all_articles(self, splits):


       
        split_count = 0 

        #for each split
        ttl_dataset = []
        for split in splits:
            
            print("Starting split:", str(split_count), "\n")
            split_count += 1

            #loop over all leanings
            for leaning in split:

                print("For leaning:", leaning.upper())
                
                if split_count == 1:
                    training_dataset = split[leaning][ApplicationConstants.Train]
                    validation_dataset = split[leaning][ApplicationConstants.Validation]
                    test_dataset = split[leaning][ApplicationConstants.Test]
                    ttl_dataset += training_dataset
                    ttl_dataset += validation_dataset
                    ttl_dataset += test_dataset
                #print(ttl_dataset)
                
        print(len(ttl_dataset))
        labels, embeddings, model = self.embed_fold(list(map(lambda article: article.Content, ttl_dataset)), list(map(lambda article: article.Label.TargetGender, ttl_dataset)), 1, "all")


    def train_all(self, splits):
        ''' trains all models against all leanings
        
        Parameters: 
        ------------
        splits: A list of the splits 

        ''' 
        models = [SVM(), KNN(), Naive_Bayes(), Linear_Classifier(), NN()]
        bP = []
        bR = []
        bF = []
        fP = []
        fR = []
        fF = []
        uP = []
        uR = []
        uF = []
        hP = []
        hR = []
        hF = []
        nP = []
        nR = []
        nF = []

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
                #training_labels, training_embeddings, mod = self.embed_fold(list(map(lambda article: article.Content, training_dataset)), list(map(lambda article: article.Label.TargetGender, training_dataset)), split_count, leaning)

                #validation embeddings 
                validation_dataset = split[leaning][ApplicationConstants.Validation]
                #validation_labels, validation_embeddings = self.embed_fold(list(map(lambda article: article.Content, validation_dataset)), list(map(lambda article: article.Label.TargetGender, validation_dataset)))
              #  validation_labels, validation_embeddings = self.docEmbed.gen_vec(mod, list(map(lambda article: article.Content, validation_dataset)), list(map(lambda article: article.Label.TargetGender, validation_dataset))) 
              #  validation_labels = list(validation_labels)
                #validation_labels = list(map(lambda article: article.Label.TargetGender, validation_dataset))
                #NEED VALIDATION LABELS

                #test embeddings
                test_dataset = split[leaning][ApplicationConstants.Test]
                
                #test_labels, test_embeddings = self.embed_fold(list(map(lambda article: article.Content, test_dataset)), list(map(lambda article: article.Label.TargetGender, test_dataset)))
              #  test_labels, test_embeddings = self.docEmbed.gen_vec(mod, list(map(lambda article: article.Content,test_dataset)), list(map(lambda article: article.Label.TargetGender, test_dataset))) 
              #  test_labels = list(test_labels)
               # print(test_labels)
                #test_labels = list(map(lambda article: article.Label.TargetGender, test_dataset))

                fucking_labels, fucking_embeddings, fucking_model = self.embed_fold(list(map(lambda article: article.Content, training_dataset + validation_dataset + test_dataset)), list(map(lambda article: article.Label.TargetGender, training_dataset + validation_dataset + test_dataset)), split_count, leaning)
                training_embeddings = fucking_embeddings[:len(training_dataset)]
                training_labels = fucking_labels[:len(training_dataset)]
                validation_embeddings = fucking_embeddings[len(training_dataset): len(training_dataset) + len(validation_dataset)]
                validation_labels = fucking_labels[len(training_dataset): len(training_dataset) + len(validation_dataset)]

                test_embeddings = fucking_embeddings[len(training_dataset) + len(validation_dataset):]
                test_labels = fucking_labels[len(training_dataset) + len(validation_dataset):]

                for model in models: 

                    #get prediction from embeddings 
                    model.Train(training_embeddings, training_labels, validation_embeddings, validation_labels)
                    prediction = model.Predict(test_embeddings)
                    '''
                    print("Model:", str(type(model)).split('.')[2].split('\'')[0], "precision:", self.Metrics.Precision(prediction, test_labels), "recall:", self.Metrics.Recall(prediction, test_labels), "F-Measure:", self.Metrics.Fmeasure(prediction, test_labels))   
                    if leaning == "breitbart":
                        bP.append(self.Metrics.Precision(prediction, test_labels))
                        bR.append(self.Metrics.Recall(prediction, test_labels))
                        bF.append(self.Metrics.Fmeasure(prediction, test_labels))
                    if leaning == "fox":
                        fP.append(self.Metrics.Precision(prediction, test_labels))
                        fR.append(self.Metrics.Recall(prediction, test_labels))
                        fF.append(self.Metrics.Fmeasure(prediction, test_labels))
                    if leaning == "usa_today":
                        uP.append(self.Metrics.Precision(prediction, test_labels))
                        uR.append(self.Metrics.Recall(prediction, test_labels))
                        uF.append(self.Metrics.Fmeasure(prediction, test_labels))
                    if leaning == "new_york_times":
                        nP.append(self.Metrics.Precision(prediction, test_labels))
                        nR.append(self.Metrics.Recall(prediction, test_labels))
                        nF.append(self.Metrics.Fmeasure(prediction, test_labels))
                    if leaning == "huffpost":
                        hP.append(self.Metrics.Precision(prediction, test_labels))
                        hR.append(self.Metrics.Recall(prediction, test_labels))
                        hF.append(self.Metrics.Fmeasure(prediction, test_labels))
                    '''


                #model = models[0] 
                #model.Model.coefs_[model.Model.n_layers_ - 2]
                if split_count == 1:
                    self.Visualizer.plot_TSNE(leaning, training_embeddings + validation_embeddings + test_embeddings, training_labels + validation_labels + test_labels, training_dataset + validation_dataset + test_dataset)
        '''
        BttlS = 0
        BttlK = 0
        BttlN = 0
        BttlL = 0
        BttlNet = 0
        FttlS = 0
        FttlK = 0
        FttlN = 0
        FttlL = 0
        FttlNet = 0
        UttlS = 0
        UttlK = 0
        UttlN = 0
        UttlL = 0
        UttlNet = 0
        HttlS = 0
        HttlK = 0
        HttlN = 0
        HttlL = 0
        HttlNet = 0
        NttlS = 0
        NttlK = 0
        NttlN = 0
        NttlL = 0
        NttlNet = 0

        
        for i in range(len(bP)):
            if i %5 == 0:
                BttlS +=bP[i]
                FttlS +=fP[i]
                UttlS += uP[i]
                HttlS += hP[i]
                NttlS += nP[i]
            if i % 5 == 1:
                BttlK +=bP[i]
                FttlK +=fP[i]
                UttlK += uP[i]
                HttlK += hP[i]
                NttlK += nP[i]
            if i % 5 == 2:
                BttlN +=bP[i]
                FttlN +=fP[i]
                UttlN += uP[i]
                HttlN += hP[i]
                NttlN += nP[i]
            if i %5 == 3:
                BttlL +=bP[i]
                FttlL +=fP[i]
                UttlL += uP[i]
                HttlL += hP[i]
                NttlL += nP[i]
            if i%5 == 4:
                BttlNet +=bP[i]
                FttlNet +=fP[i]
                UttlNet += uP[i]
                HttlNet += hP[i]
                NttlNet += nP[i]
        bp = bP
        print("Precisions- Breitbart SVM: " + str(BttlS /(len(bP)/5)) + "Breitbart KNN:" + str(BttlK/(len(bP)/5)) + "Breitbart NB:" + str(BttlN /(len(bP)/5)) + "Breitbart LC: " +str(BttlL /(len(bP)/5)) + "Breitbart NN:" + str(BttlNet/(len(bP)/5)))
        print("Precisions- Fox SVM: " + str(FttlS /(len(bP)/5)) + "Fox KNN:" + str(FttlK/(len(bP)/5)) + "Fox NB:" + str(FttlN /(len(bP)/5)) + "Fox LC: " +str(FttlL /(len(bP)/5)) + "Fox NN:" + str(FttlNet/(len(bP)/5)))
        print("Precisions- USA SVM: " + str(UttlS /(len(bP)/5)) + "USA KNN:" + str(UttlK/(len(bP)/5)) + "USA NB:" + str(UttlN /(len(bP)/5)) + "USA LC: " +str(UttlL /(len(bP)/5)) + "USA NN:" + str(UttlNet/(len(bP)/5)))
        print("Precisions- Huffpost SVM: " + str(HttlS /(len(bP)/5)) + "Huffpost KNN:" + str(HttlK/(len(bp)/5)) + "Huffpost NB:" + str(HttlN /(len(bp)/5)) + "Huffpost LC: " +str(HttlL /(len(bp)/5)) + "Huffpost NN:" + str(HttlNet/(len(bp)/5)))
        print("Precisions- NYT SVM: " + str(NttlS /(len(bP)/5)) + "NYT KNN:" + str(NttlK/(len(bp)/5)) + "NYT NB:" + str(NttlN /(len(bp)/5)) + "NYT LC: " +str(NttlL /(len(bp)/5)) + "NYT NN:" + str(NttlNet/(len(bp)/5)))
    
        BttlS = 0
        BttlK = 0
        BttlN = 0
        BttlL = 0
        BttlNet = 0
        FttlS = 0
        FttlK = 0
        FttlN = 0
        FttlL = 0
        FttlNet = 0
        UttlS = 0
        UttlK = 0
        UttlN = 0
        UttlL = 0
        UttlNet = 0
        HttlS = 0
        HttlK = 0
        HttlN = 0
        HttlL = 0
        HttlNet = 0
        NttlS = 0
        NttlK = 0
        NttlN = 0
        NttlL = 0
        NttlNet = 0

        bp = bP
        for i in range(len(bR)):
            if i %5 == 0:
                BttlS +=bR[i]
                FttlS +=fR[i]
                UttlS += uR[i]
                HttlS += hR[i]
                NttlS += nR[i]
            if i % 5 == 1:
                BttlK +=bR[i]
                FttlK +=fR[i]
                UttlK += uR[i]
                HttlK += hR[i]
                NttlK += nR[i]
            if i % 5 == 2:
                BttlN +=bR[i]
                FttlN +=fR[i]
                UttlN += uR[i]
                HttlN += hR[i]
                NttlN += nR[i]
            if i %5 == 3:
                BttlL +=bR[i]
                FttlL +=fR[i]
                UttlL += uR[i]
                HttlL += hR[i]
                NttlL += nR[i]
            if i%5 == 4:
                BttlNet +=bR[i]
                FttlNet +=fR[i]
                UttlNet += uR[i]
                HttlNet += hR[i]
                NttlNet += nR[i]
        print("Recalls- Breitbart SVM: " + str(BttlS /(len(bP)/5)) + "Breitbart KNN:" + str(BttlK/(len(bp)/5)) + "Breitbart NB:" + str(BttlN /(len(bp)/5)) + "Breitbart LC: " +str(BttlL /(len(bp)/5)) + "Breitbart NN:" + str(BttlNet/(len(bp)/5)))
        print("Recalls- Fox SVM: " + str(FttlS /(len(bP)/5)) + "Fox KNN:" + str(FttlK/(len(bp)/5)) + "Fox NB:" + str(FttlN /(len(bp)/5)) + "Fox LC: " +str(FttlL /(len(bp)/5)) + "Fox NN:" + str(FttlNet/(len(bp)/5)))
        print("Recalls- USA SVM: " + str(UttlS /(len(bP)/5)) + "USA KNN:" + str(UttlK/(len(bp)/5)) + "USA NB:" + str(UttlN /(len(bp)/5)) + "USA LC: " +str(UttlL /(len(bp)/5)) + "USA NN:" + str(UttlNet/(len(bp)/5)))
        print("Recalls- Huffpost SVM: " + str(HttlS /(len(bP)/5)) + "Huffpost KNN:" + str(HttlK/(len(bp)/5)) + "Huffpost NB:" + str(HttlN /(len(bp)/5)) + "Huffpost LC: " +str(HttlL /(len(bp)/5)) + "Huffpost NN:" + str(HttlNet/(len(bp)/5)))
        print("Recalls- NYT SVM: " + str(NttlS /(len(bP)/5)) + "NYT KNN:" + str(NttlK/(len(bp)/5)) + "NYT NB:" + str(NttlN /(len(bp)/5)) + "NYT LC: " +str(NttlL /(len(bp)/5)) + "NYT NN:" + str(NttlNet/(len(bp)/5)))
    

        BttlS = 0
        BttlK = 0
        BttlN = 0
        BttlL = 0
        BttlNet = 0
        FttlS = 0
        FttlK = 0
        FttlN = 0
        FttlL = 0
        FttlNet = 0
        UttlS = 0
        UttlK = 0
        UttlN = 0
        UttlL = 0
        UttlNet = 0
        HttlS = 0
        HttlK = 0
        HttlN = 0
        HttlL = 0
        HttlNet = 0
        NttlS = 0
        NttlK = 0
        NttlN = 0
        NttlL = 0
        NttlNet = 0

        bp = bP
        for i in range(len(bR)):
            if i %5 == 0:
                BttlS +=bF[i]
                FttlS +=fF[i]
                UttlS += uF[i]
                HttlS += hF[i]
                NttlS += nF[i]
            if i % 5 == 1:
                BttlK +=bF[i]
                FttlK +=fF[i]
                UttlK += uF[i]
                HttlK += hF[i]
                NttlK += nF[i]
            if i % 5 == 2:
                BttlN +=bF[i]
                FttlN +=fF[i]
                UttlN += uF[i]
                HttlN += hF[i]
                NttlN += nF[i]
            if i %5 == 3:
                BttlL +=bF[i]
                FttlL +=fF[i]
                UttlL += uF[i]
                HttlL += hF[i]
                NttlL += nF[i]
            if i%5 == 4:
                BttlNet +=bF[i]
                FttlNet +=fF[i]
                UttlNet += uF[i]
                HttlNet += hF[i]
                NttlNet += nF[i]
        print("F1- Breitbart SVM: " + str(BttlS /(len(bP)/5)) + "Breitbart KNN:" + str(BttlK/(len(bp)/5)) + "Breitbart NB:" + str(BttlN /(len(bp)/5)) + "Breitbart LC: " +str(BttlL /(len(bp)/5)) + "Breitbart NN:" + str(BttlNet/(len(bp)/5)))
        print("F1- Fox SVM: " + str(FttlS /(len(bP)/5)) + "Fox KNN:" + str(FttlK/(len(bp)/5)) + "Fox NB:" + str(FttlN /(len(bp)/5)) + "Fox LC: " +str(FttlL /(len(bp)/5)) + "Fox NN:" + str(FttlNet/(len(bp)/5)))
        print("F1- USA SVM: " + str(UttlS /(len(bP)/5)) + "USA KNN:" + str(UttlK/(len(bp)/5)) + "USA NB:" + str(UttlN /(len(bp)/5)) + "USA LC: " +str(UttlL /(len(bp)/5)) + "USA NN:" + str(UttlNet/(len(bp)/5)))
        print("F1- Huffpost SVM: " + str(HttlS /(len(bP)/5)) + "Huffpost KNN:" + str(HttlK/(len(bp)/5)) + "Huffpost NB:" + str(HttlN /(len(bp)/5)) + "Huffpost LC: " +str(HttlL /(len(bp)/5)) + "Huffpost NN:" + str(HttlNet/(len(bp)/5)))
        print("F1- NYT SVM: " + str(NttlS /(len(bP)/5)) + "NYT KNN:" + str(NttlK/(len(bp)/5)) + "NYT NB:" + str(NttlN /(len(bp)/5)) + "NYT LC: " +str(NttlL /(len(bp)/5)) + "NYT NN:" + str(NttlNet/(len(bp)/5)))
        '''    


orchestrator = Orchestrator()
splits = orchestrator.read_data(clean=False, number_of_articles=25) 
#print("Dirty .25")
#orchestrator.run_sentiment_analysis_all(splits[0]) 
#orchestrator.train_all(splits)
#orchestrator.embed_all_articles(splits)
imdb_vec, imdb_labels = orchestrator.imdb()
orchestrator.train_sent_models(imdb_vec, imdb_labels)






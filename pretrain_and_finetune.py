'''This file will perform doc2vec pretrain on atn and finetune on newsbias, then use the doc2vec embeddings to
generate a Neural Network Model, and TSNE visualizations
'''
from DataReader import DataReader
from DataContracts import Article
from doc2vec import doc
from Metrics import Metrics
from Visualizer import Visualizer

import ApplicationConstants
from DataContracts import Article

import re

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from Models.NN_engine import NN

import numpy as np
from preprocessor import Preprocessor

import os.path
import copy
import time
import random
import pickle

class pretrain():

    def __init__(self):

        self.Preprocessor = Preprocessor()
        self.Visualizer = Visualizer()
        self. breitbartTtlPrecision, self.foxTtlPrecision, self.usaTtlPrecision, self.huffTtlPrecision, self.nytTtlPrecision = 0, 0, 0, 0, 0
        self.breitbartTtlRecall, self.foxTtlRecall, self.usaTtlRecall, self.huffTtlRecall, self.nytTtlRecall = 0, 0, 0, 0, 0
        self.breitbartTtlF1, self.foxTtlF1, self.usaTtlF1, self.huffTtlF1, self.nytTtlF1 = 0, 0, 0, 0, 0

    def filter_ATN_content(self, content, publication=None):
        content = re.sub("â€œ|â€\?|“|”|\"\"", '"', content)
        content = re.sub("â€œ|â€\?|” | ”|\"\"", '"', content)
        content = re.sub('â€"', '—', content)
        content = content.lower()
        content = re.sub('(?:https?:)?//[\w\d]+\.[\w\d/\.]+', '', content)
        content = re.sub('[\w\d/\.]+(?:\.com|\.net|\.org|\.co)[\w\d/\.]*', '', content)
        content = re.sub(' {2,}', ' ', content)
        content = re.sub(r"#(\w+)", '', content)
        content = re.sub(r"@(\w+)", '', content)
        content = re.sub("reuters", '', content)
        content = re.sub("(reuters)", '', content)
        content = re.sub("\\/", ' ', content)
        content = re.sub("(?<=/)[^/]+(?=/)", ' ', content)
        content = re.sub("\\n", '', content)
        content = re.sub("\\'s", '\'s', content)
        content = re.sub("\\'t", '\'t', content)
        content = re.sub("\\'d", '\'d', content)
        content = re.sub("\\'re", '\'re', content)
        content = re.sub("\\\'", '\'', content)
        content = re.sub("\\xa0", ' ', content)
        content = re.sub("\(\)", '', content)
        content = re.sub(" ing ", '', content)

        return content
    def pretrain_and_fineTune(self, atnPortion = 0.2, dirtyNewsBias=True, cleanatn = True):
        reader = DataReader()  # adds 2 G
        if os.path.exists("./store/") == False:
            os.mkdir("./store/")
        portionToLoad = atnPortion
        print("Dirty Finetune: ", dirtyNewsBias, " Clean Atn", cleanatn)
        if os.path.exists("./metrics/") == False:
            os.mkdir("./metrics/")
        if os.path.exists("./PretrainFinetuneStorage/") == False:
            os.mkdir("./PretrainFinetuneStorage/")

        if (os.path.exists('store/model_pretrained_cleaned_atn.model')) == False and cleanatn:
            if atnPortion == 0.2:
                appConst = ApplicationConstants.all_the_news_cleaned_path_20
                if (os.path.exists(appConst)) == False:
                    print("The all the news dataset needs to be cleaned. This will take a LONG time...")
                    time.sleep(10)  #pause to allow to see message
                    reader.Load_ATN_csv(0.20, clean=True, save=True)
            elif atnPortion == 1.0: #note: ALL OF THE DATA REQUIRES A HUGE AMOUNT OF RAM- we used .2 for our exp
                appConst = ApplicationConstants.all_the_news_cleaned_path_all
                if (os.path.exists(appConst)) == False:
                    print("The all the news dataset needs to be cleaned. This will take a LONG time...")
                    time.sleep(10) #pause to allow to see message
                    reader.Load_ATN_csv(1.0, clean=True, save=True)
            print("Cleaned pretrained atn model does not exist, loading data")
            all_the_news = reader.Load_newer_ATN(appConst, 1.0)
            print("Number articles loaded: ", str(len(all_the_news)))
            cleaned_articles = list(map(lambda article: article.Content, all_the_news))
            pretrain_labels = list(map(lambda article: article.Label, all_the_news))
            del all_the_news #delete the object


        elif (os.path.exists('store/model_pretrained_dirty_atn.model')) == False and cleanatn == False: #load the dirty atn
            print("Dirty pretrained atn model does not exist, loading data")
            print("Loading %.2f All The News" % portionToLoad)
            all_the_news = reader.Load_newer_ATN(ApplicationConstants.all_the_news_newer_path, portionToLoad)

            dirty_pretrain_content = list(map(lambda article: article.Content, all_the_news))

            cleaned_articles = []
            print("Removing twitter tags and emails from All The News")
            for article in dirty_pretrain_content:
                cleaned_articles.append(self.filter_ATN_content(article)) #only remove twitter tags (not full clean)


            dirty_pretrain_content.clear()
            del dirty_pretrain_content #delete the object
            pretrain_labels = list(map(lambda article: article.Label, all_the_news))
            del all_the_news# these values are null since ATN doesn't have gender labels


        del reader

        pretrain_epoch = 25
        fineTune_epoch = 100
        vector_size = 100
        if cleanatn:
            if dirtyNewsBias == False:
                avFile = "metrics/cleanPretrainATN_cleanFineTuneNewsBias_averages.txt"  # "pretrain_and_fineTune_atnClean_av.txt"
                allfile = "metrics/cleanPretrainATN_cleanFineTuneNewsBias_byFold.txt"  # "pretrain_and_fineTune_atnClean.txt"
            else:
                avFile = "metrics/cleanPretrainATN_dirtyFineTuneNewsBias_averages.txt"  # "pretrain_and_fineTune_atnClean_av.txt"
                allfile = "metrics/cleanPretrainATN_dirtyFineTuneNewsBias_byFold.txt"  # "pretrain_and_fineTune_atnClean.txt"
        else:
            if dirtyNewsBias == False:
                avFile = "metrics/dirtyPretrainATN_cleanFineTuneNewsBias_averages.txt"  # "pretrain_and_fineTune_atnClean_av.txt"
                allfile = "metrics/dirtyPretrainATN_cleanFineTuneNewsBias_byFold.txt"  # "pretrain_and_fineTune_atnClean.txt"
            else:
                avFile = "metrics/dirtyPretrainATN_dirtyFineTuneNewsBias_averages.txt"  # "pretrain_and_fineTune_atnClean_av.txt"
                allfile = "metrics/dirtyPretrainATN_dirtyFineTuneNewsBias_byFold.txt"  # "pretrain_and_fineTune_atnClean.txt"




        with open(avFile, "a+") as f_av:
            with open(allfile, "a+") as f:
                print("PRETRAINING DOC2VEC MODEL")
                time.sleep(10)  # pause to allow to see message

                docEmbed = doc()
                if cleanatn ==False:
                    if (os.path.exists('store/model_pretrained_dirty_atn.model')):
                        print("loading uncleaned atn model")
                        pretrained_article_model = docEmbed.Load_Model('store/model_pretrained_dirty_atn.model')
                    else:
                        print("Doing pretrain and generating uncleaned atn model... This will take a while.")
                        pretrained_article_model = docEmbed.Embed(cleaned_articles, pretrain_labels,
                                                                       vector_size=vector_size,
                                                                       epochs=pretrain_epoch)
                        pretrained_article_model.save('store/model_pretrained_dirty_atn.model')
                else:
                    if (os.path.exists('store/model_pretrained_cleaned_atn.model')):
                        print("loading cleaned atn model")
                        pretrained_article_model = docEmbed.Load_Model('store/model_pretrained_cleaned_atn.model')
                    else:
                        print("Doing pretrain and generating cleaned atn model... This will take a while. ")
                        pretrained_article_model = docEmbed.Embed(cleaned_articles, pretrain_labels,
                                                                       vector_size=vector_size,
                                                                       epochs=pretrain_epoch)
                        print("saving cleaned embed atn")
                        pretrained_article_model.save('store/model_pretrained_cleaned_atn.model')

                del docEmbed
                reader = DataReader()
                if dirtyNewsBias:
                    time.sleep(10)  # pause to allow to see message
                    finetuneSet = reader.Load_Splits(ApplicationConstants.all_articles_random_v4, None,
                                                     number_of_articles=50,
                                                     clean=False, save=False, shouldRandomize=False)

                else:
                    time.sleep(10)  # pause to allow to see message
                    finetuneSet = reader.Load_Splits(ApplicationConstants.all_articles_random_v4_cleaned, None,
                                                     number_of_articles=50,
                                                     clean=False, save=False, shouldRandomize=False)
                del reader

                print("FINE TUNING DOC2VEC MODEL SEPARATELY ON EACH FOLD")

                print("Training vector size " + str(vector_size) + " pretrain " + str(
                    pretrain_epoch) + " finetune " + str(fineTune_epoch))

                for i, split in enumerate(finetuneSet):
                    foldNum = str(i+1)
                    print("Fold " + str (i+1))
                    for j, leaning in enumerate(split):
                        training_dataset = split[leaning][ApplicationConstants.Train]
                        validation_dataset = split[leaning][ApplicationConstants.Validation]
                        test_dataset = split[leaning][ApplicationConstants.Test]

                        fineTune_train_articles = list(
                            map(lambda article: article.Content, training_dataset))
                        fineTune_train_labels = list(
                            map(lambda article: article.Label.TargetGender, training_dataset))
                        fineTune_val_articles = list(
                            map(lambda article: article.Content, validation_dataset))
                        fineTune_val_labels = list(
                            map(lambda article: article.Label.TargetGender, validation_dataset))
                        fineTune_test_articles = list(map(lambda article: article.Content, test_dataset))
                        fineTune_test_labels = list(map(lambda article: article.Label.TargetGender, test_dataset))

                        docEmbed = doc()

                        fine_tuned_model = docEmbed.fine_tune(fineTune_train_articles ,fineTune_train_labels,
                                                                    pretrained_article_model, fineTune_epoch)

                        #One model per fold, since they're trained on different people
                        if dirtyNewsBias == False and cleanatn == True:
                            fine_tuned_model.save('PretrainFinetuneStorage/model_pretrainedCleanATN_finetunedClean_fold' + str(foldNum) + '_' + leaning + '.model')
                        elif dirtyNewsBias == False and cleanatn == False:
                            fine_tuned_model.save('PretrainFinetuneStorage/model_pretrainedDirtyATN_finetunedClean_fold' + str(foldNum) + '_' + leaning +'.model')
                        elif dirtyNewsBias == True and cleanatn == True:
                            fine_tuned_model.save(
                                'PretrainFinetuneStorage/model_pretrainedCleanATN_finetunedDirty_fold' + str(foldNum) + '_' + leaning + '.model')
                        elif dirtyNewsBias == True and cleanatn == False:
                            fine_tuned_model.save(
                                'PretrainFinetuneStorage/model_pretrainedDirtyATN_finetunedDirty_fold' + str(foldNum) + '_' + leaning + '.model')

                        FT_Train_labels, FT_train_embeddings = docEmbed.gen_vec(fine_tuned_model,
                                                                             fineTune_train_articles,
                                                                              fineTune_train_labels)

                        FT_Train_labels, FT_train_embeddings = self.shuffle(FT_Train_labels, FT_train_embeddings)

                        FT_Val_labels, FT_val_embeddings = docEmbed.gen_vec(fine_tuned_model,
                                                                             fineTune_val_articles,
                                                                              fineTune_val_labels)

                        FT_Val_labels, FT_val_embeddings = self.shuffle(FT_Val_labels, FT_val_embeddings)

                        FT_Test_labels, TF_Test_embeddings = docEmbed.gen_vec(fine_tuned_model,
                                                                              fineTune_test_articles,
                                                                            fineTune_test_labels)
                        FT_Test_labels, TF_Test_embeddings = self.shuffle(FT_Test_labels, TF_Test_embeddings)

                        FT_labels = list(FT_Train_labels)
                        FT_val_labels = list(FT_Val_labels)
                        FT_test_labels = list(FT_Test_labels) #test labels
                        del docEmbed
                        maxF1 = 0
                        print("Training and validating the NN 10 times on the data for the fold/outlet, and taking the best model")
                        for k in range(10):
                            model = NN()

                            model.Train(FT_train_embeddings, FT_labels, FT_val_embeddings, FT_val_labels) #val isn't actually done here

                            prediction = model.Predict(FT_val_embeddings) #val done here

                            Met = Metrics()
                            F1 = self.metric_calculation(j, prediction, FT_labels, training_dataset, validation_dataset, Met, FT_val_labels, f, file_write = False)
                            if F1 > maxF1:
                                maxF1 = F1

                                filename= "PretrainFinetuneStorage/" + leaning + "_" + str(i) + "_NN.sav"
                                pickle.dump(model, open(filename, 'wb'))
                            del Met
                        with open(filename, 'rb') as pickleF:
                            best_model = pickle.load(pickleF)
                        print("Taking the best model for the fold/outlet and getting the test metrics")
                        prediction = best_model.Predict(TF_Test_embeddings)
                        Met = Metrics()
                        self.metric_calculation(j,  prediction, FT_labels, training_dataset, test_dataset, Met, FT_test_labels, f, file_write = True) #write score to file with best model & store in class vars

                        if j == 0:
                            lean = "Breitbart"
                        if j == 1:
                            lean = "Fox"
                        if j == 2:
                            lean = "USA"
                        if j == 3:
                            lean = "HuffPost"
                        if j == 4:
                            lean = "NYT"
                        self.decide_visual(prediction, FT_test_labels, leaning, lean, split,  Met, fine_tuned_model, dirty = dirtyNewsBias) #do vis if necessary


                self.av_calculation(f_av)


    def shuffle(self, FT_labels, FT_embed):
        shuffledList = list(zip(FT_labels, FT_embed))
        random.shuffle(shuffledList)
        return zip (*shuffledList)

    def av_calculation(self, f_av):
        f_av.write("Average Breitbart Recall: " + str(self.breitbartTtlRecall / 5) + " Average Breitbart Precision: " + str(
            self.breitbartTtlPrecision / 5) + " Average Breitbart F1: " + str(self.breitbartTtlF1 / 5) + "\n")
        f_av.write("Average Fox Recall: " + str(
            self.foxTtlRecall / 5) + " Average Fox Precision: " + str(
            self.foxTtlPrecision / 5) + " Average Fox F1: " + str(self.foxTtlF1 / 5) + "\n")
        f_av.write("Average USA Recall: " + str(
            self.usaTtlRecall / 5) + " Average USA Precision: " + str(
            self.usaTtlPrecision / 5) + " Average USA F1: " + str(self.usaTtlF1 / 5) + "\n")
        f_av.write("Average Huffpost Recall: " + str(
            self.huffTtlRecall / 5) + " Average Huffpost Precision: " + str(
            self.huffTtlPrecision / 5) + " Average Huffpost F1: " + str(self.huffTtlF1 / 5) + "\n")
        f_av.write("Average NYT Recall: " + str(
            self.nytTtlRecall / 5) + " Average NYT Precision: " + str(
            self.nytTtlPrecision / 5) + " Average NYT F1: " + str(self.nytTtlF1 / 5) + "\n")

        print("Average Breitbart Recall: " + str(
            self.breitbartTtlRecall / 5) + " Average Breitbart Precision: " + str(
            self.breitbartTtlPrecision / 5) + " Average Breitbart F1: " + str(self.breitbartTtlF1 / 5))
        print("Average Fox Recall: " + str(
            self.foxTtlRecall / 5) + " Average Fox Precision: " + str(
            self.foxTtlPrecision / 5) + " Average Fox F1: " + str(self.foxTtlF1 / 5))
        print("Average USA Recall: " + str(
            self.usaTtlRecall / 5) + " Average USA Precision: " + str(
            self.usaTtlPrecision / 5) + " Average USA F1: " + str(self.usaTtlF1 / 5))
        print("Average Huffpost Recall: " + str(
            self.huffTtlRecall / 5) + " Average Huffpost Precision: " + str(
            self.huffTtlPrecision / 5) + " Average Huffpost F1: " + str(self.huffTtlF1 / 5))
        print("Average NYT Recall: " + str(
            self.nytTtlRecall / 5) + " Average NYT Precision: " + str(
            self.nytTtlPrecision / 5) + " Average NYT F1: " + str(self.nytTtlF1 / 5))

    def metric_calculation(self, j, prediction, FT_labels, training_dataset, validation_dataset, Met, FT_test_labels, f, file_write = True):
        prec = Met.Precision(prediction, FT_test_labels)
        recall = Met.Recall(prediction, FT_test_labels)
        F1 = Met.Fmeasure(prediction, FT_test_labels)

        if file_write:
            if j == 0:
                lean = "Breitbart"
                print(len(prediction), len(FT_labels), len(training_dataset) + len(validation_dataset))
                self.breitbartTtlPrecision +=prec
                self.breitbartTtlRecall += recall
                self.breitbartTtlF1 += F1
            if j == 1:
                lean = "Fox"
                self.foxTtlPrecision += prec
                self.foxTtlRecall += recall
                self.foxTtlF1 += F1
            if j == 2:
                lean = "USA"
                self.usaTtlPrecision += prec
                self.usaTtlRecall += recall
                self.usaTtlF1 += F1
            if j == 3:
                lean = "Huff"
                self.huffTtlPrecision += prec
                self.huffTtlRecall += recall
                self.huffTtlF1 += F1
            if j == 4:
                lean = "NYT"
                self.nytTtlPrecision += prec
                self.nytTtlRecall += recall
                self.nytTtlF1 += F1
            print("Leaning: ", lean, " precision: ",
                  Met.Precision(prediction, FT_test_labels), " recall: ",
                  Met.Recall(prediction, FT_test_labels), " F-Measure: ",
                  Met.Fmeasure(prediction, FT_test_labels))
            f.write("Leaning: " + lean + " precision: " +
                    str(Met.Precision(prediction, FT_test_labels)) + " recall: " +
                    str(Met.Recall(prediction, FT_test_labels)) + " F-Measure: " +
                    str(Met.Fmeasure(prediction, FT_test_labels)) + "\n")

        if file_write == False:
            if j == 0:
                lean = "Breitbart"
            if j == 1:
                lean = "Fox"
            if j == 2:
                lean = "USA"
            if j == 3:
                lean = "HuffPost"
            if j == 4:
                lean = "NYT"
            print("Leaning: ", lean, " precision: ",
              Met.Precision(prediction, FT_test_labels), " recall: ",
              Met.Recall(prediction, FT_test_labels), " F-Measure: ",
              Met.Fmeasure(prediction, FT_test_labels))
            return F1

    def decide_visual(self, prediction, FT_test_labels, leaning, lean, split,  Met, fine_tuned_model, dirty = True):
        if Met.Fmeasure(prediction, FT_test_labels) > 0.70 and Met.Fmeasure(prediction, FT_test_labels) < 0.90:
            if dirty == False:
                fname = "visualizations/" + leaning +"_finetuned_cleaned.png"
            else:
                fname = "visualizations/" + leaning +"_finetuned_dirty.png"
            print(fname)
            print(leaning)
            if (os.path.exists(fname)) == False:
                if lean == 'Breitbart':
                    self.visualize_all(fine_tuned_model, fname, leaning, split)
                if lean == 'Fox':
                    self.visualize_all(fine_tuned_model, fname, leaning, split)
                if lean == 'USA':
                    self.visualize_all(fine_tuned_model, fname, leaning, split)
                if lean == 'HuffPost':
                    self.visualize_all(fine_tuned_model,fname,  leaning, split)
                if lean == 'NYT':
                    self.visualize_all(fine_tuned_model, fname, leaning, split)
        del Met

    def visualize_all(self, fine_tuned_model, fname, leaning, split):
        from Orchestrator import Orchestrator
        orchestrator = Orchestrator()
        articles = orchestrator.read_data(path=ApplicationConstants.all_articles_random_v4_cleaned,
                                          number_of_articles=1000, save=False, clean=False, random=False)
        training_dataset = split[leaning][ApplicationConstants.Train]
        validation_dataset = split[leaning][ApplicationConstants.Validation]
        test_dataset = split[leaning][ApplicationConstants.Test]
        articles = list(map(lambda article: article.Content, training_dataset + validation_dataset + test_dataset))
        labels = list(map(lambda article: article.Label.TargetGender, training_dataset + validation_dataset + test_dataset))
        docEmbed = doc()
        labels, embeddings = docEmbed.gen_vec(fine_tuned_model, articles, labels)
        labels = list(labels)
        self.Visualizer.plot_TSNE(leaning, embeddings, labels, training_dataset + validation_dataset + test_dataset, fname)




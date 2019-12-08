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

	def read_data(self, path, clean=True, save=False, number_of_articles = 50):       
		return self.Reader.Load_Splits(path, clean=clean, save=save, number_of_articles=number_of_articles)

	def imdb(self, model, label_path, vector_path):
		sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS' }
		sentences = LabeledLineSentence(sources)
		vectors, labels = sentences.generate_imdb_vec(model, label_path, vector_path)
		return vectors, labels

	def train_sent_models(self, all_articles, all_labels, leanings, article_doc2vec_label_path, article_doc2vec_vector_path, article_doc2vec_model_path, imdb_label_path, imdb_vector_path):


		if (not os.path.exists(article_doc2vec_model_path)):
			all_articles_model = self.docEmbed.Embed(all_articles, all_labels) 
			all_articles_model.save(article_doc2vec_model_path)
		else:
			all_articles_model = self.docEmbed.Load_Model(article_doc2vec_model_path) 

		if (not os.path.exists(article_doc2vec_label_path) or not os.path.exists(article_doc2vec_vector_path)):

			all_articles_labels, all_articles_vectors = self.docEmbed.gen_vec(all_articles_model, all_articles, all_labels)
			np.save(article_doc2vec_label_path, all_articles_labels)
			np.save(article_doc2vec_vector_path, all_articles_vectors)
			
		else:

			all_articles_labels = np.load(article_doc2vec_label_path)
			all_articles_vectors = np.load(article_doc2vec_vector_path)			

		imdb_vec, imdb_labels = self.imdb(all_articles_model, imdb_label_path, imdb_vector_path)

		models = [SVM()]

		for model in models:

			male = []
			female = [] 

			model.Train(imdb_vec, imdb_labels, None, None)
			predictions, confidences = model.Predict(all_articles_vectors)
			
			for index, prediction in enumerate(predictions):
									  
				if int(all_labels[index]) == ApplicationConstants.female_value:
					female.append((leanings[index], prediction, confidences[index]))
				else:
					male.append((leanings[index], prediction, confidences[index]))

			self.Visualizer.graph_sentiment(female, male)

	#def print_shit(self, fileName, allF, allM, conf):
	def print_shit(self, fileName, allF, allM):
		file = open(fileName, 'w')
		print('FEMALE\n', file = file)
		print(allF, file = file)
		print('\nMALE\n',  file = file)
		print(allM,  file = file)
		#print('\nPROBABILITIES\n',  file = file)
		#print(conf, file=file)


	def embed_fold(self, articles, labels, fold, leaning):
		''' 
		trains and returns the vector embeddings for doc2vec or sent2vec 

		Parameters:5
		articles: a list of articles that are cleaned
		labels: a list of labels corresponding to the article genders
		''' 

		#emb = self.docEmbed.word2vec() 
		model = self.docEmbed.Embed(articles, labels)
		targets, regressors = self.docEmbed.gen_vec(model, articles, labels)

		return list(targets), regressors, model
	
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

				#validation embeddings 
				validation_dataset = split[leaning][ApplicationConstants.Validation]

				#test embeddings
				test_dataset = split[leaning][ApplicationConstants.Test]           

				article_labels, article_embeddings, article_model = self.embed_fold(list(map(lambda article: article.Content, training_dataset + validation_dataset + test_dataset)), list(map(lambda article: article.Label.TargetGender, training_dataset + validation_dataset + test_dataset)), split_count, leaning)
				training_embeddings = article_embeddings[:len(training_dataset)]
				training_labels = article_labels[:len(training_dataset)]
				validation_embeddings = article_embeddings[len(training_dataset): len(training_dataset) + len(validation_dataset)]
				validation_labels = article_labels[len(training_dataset): len(training_dataset) + len(validation_dataset)]

				test_embeddings = article_embeddings[len(training_dataset) + len(validation_dataset):]
				test_labels = article_labels[len(training_dataset) + len(validation_dataset):]

				for model in models: 

					#get prediction from embeddings 
					model.Train(training_embeddings, training_labels, validation_embeddings, validation_labels)
					prediction, confidence = model.Predict(test_embeddings)
					
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
					


				#model = models[0] 
				#model.Model.coefs_[model.Model.n_layers_ - 2]
				if split_count == 1:
					self.Visualizer.plot_TSNE(leaning, training_embeddings + validation_embeddings + test_embeddings, training_labels + validation_labels + test_labels, training_dataset + validation_dataset + test_dataset)
		
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
		print("Precisions- Breitbart SVM: " + str(BttlS /(len(bP)/5)) + " Breitbart KNN: " + str(BttlK/(len(bP)/5)) + " Breitbart NB: " + str(BttlN /(len(bP)/5)) + " Breitbart LC: " +str(BttlL /(len(bP)/5)) + " Breitbart NN: " + str(BttlNet/(len(bP)/5)))
		print("Precisions- Fox SVM: " + str(FttlS /(len(bP)/5)) + " Fox KNN: " + str(FttlK/(len(bP)/5)) + " Fox NB: " + str(FttlN /(len(bP)/5)) + " Fox LC: " +str(FttlL /(len(bP)/5)) + " Fox NN: " + str(FttlNet/(len(bP)/5)))
		print("Precisions- USA SVM: " + str(UttlS /(len(bP)/5)) + " USA KNN: " + str(UttlK/(len(bP)/5)) + " USA NB: " + str(UttlN /(len(bP)/5)) + " USA LC: " +str(UttlL /(len(bP)/5)) + " USA NN: " + str(UttlNet/(len(bP)/5)))
		print("Precisions- Huffpost SVM: " + str(HttlS /(len(bP)/5)) + " Huffpost KNN: " + str(HttlK/(len(bp)/5)) + " Huffpost NB: " + str(HttlN /(len(bp)/5)) + " Huffpost LC: " +str(HttlL /(len(bp)/5)) + " Huffpost NN: " + str(HttlNet/(len(bp)/5)))
		print("Precisions- NYT SVM: " + str(NttlS /(len(bP)/5)) + " NYT KNN: " + str(NttlK/(len(bp)/5)) + " NYT NB: " + str(NttlN /(len(bp)/5)) + " NYT LC: " +str(NttlL /(len(bp)/5)) + " NYT NN: " + str(NttlNet/(len(bp)/5)))
	
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
		print("Recalls- Breitbart SVM: " + str(BttlS /(len(bP)/5)) + " Breitbart KNN: " + str(BttlK/(len(bp)/5)) + " Breitbart NB: " + str(BttlN /(len(bp)/5)) + " Breitbart LC: " +str(BttlL /(len(bp)/5)) + " Breitbart NN: " + str(BttlNet/(len(bp)/5)))
		print("Recalls- Fox SVM: " + str(FttlS /(len(bP)/5)) + " Fox KNN: " + str(FttlK/(len(bp)/5)) + " Fox NB: " + str(FttlN /(len(bp)/5)) + " Fox LC: " +str(FttlL /(len(bp)/5)) + " Fox NN: " + str(FttlNet/(len(bp)/5)))
		print("Recalls- USA SVM: " + str(UttlS /(len(bP)/5)) + " USA KNN: " + str(UttlK/(len(bp)/5)) + " USA NB: " + str(UttlN /(len(bp)/5)) + " USA LC: " +str(UttlL /(len(bp)/5)) + " USA NN: " + str(UttlNet/(len(bp)/5)))
		print("Recalls- Huffpost SVM: " + str(HttlS /(len(bP)/5)) + " Huffpost KNN: " + str(HttlK/(len(bp)/5)) + " Huffpost NB: " + str(HttlN /(len(bp)/5)) + " Huffpost LC: " +str(HttlL /(len(bp)/5)) + " Huffpost NN: " + str(HttlNet/(len(bp)/5)))
		print("Recalls- NYT SVM: " + str(NttlS /(len(bP)/5)) + " NYT KNN: " + str(NttlK/(len(bp)/5)) + " NYT NB: " + str(NttlN /(len(bp)/5)) + " NYT LC: " +str(NttlL /(len(bp)/5)) + " NYT NN: " + str(NttlNet/(len(bp)/5)))
	

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
		print("F1- Breitbart SVM: " + str(BttlS /(len(bP)/5)) + " Breitbart KNN: " + str(BttlK/(len(bp)/5)) + " Breitbart NB: " + str(BttlN /(len(bp)/5)) + " Breitbart LC: " +str(BttlL /(len(bp)/5)) + " Breitbart NN: " + str(BttlNet/(len(bp)/5)))
		print("F1- Fox SVM: " + str(FttlS /(len(bP)/5)) + " Fox KNN: " + str(FttlK/(len(bp)/5)) + " Fox NB: " + str(FttlN /(len(bp)/5)) + " Fox LC: " +str(FttlL /(len(bp)/5)) + " Fox NN: " + str(FttlNet/(len(bp)/5)))
		print("F1- USA SVM: " + str(UttlS /(len(bP)/5)) + " USA KNN: " + str(UttlK/(len(bp)/5)) + " USA NB: " + str(UttlN /(len(bp)/5)) + " USA LC: " +str(UttlL /(len(bp)/5)) + " USA NN: " + str(UttlNet/(len(bp)/5)))
		print("F1- Huffpost SVM: " + str(HttlS /(len(bP)/5)) + " Huffpost KNN: " + str(HttlK/(len(bp)/5)) + " Huffpost NB: " + str(HttlN /(len(bp)/5)) + " Huffpost LC: " +str(HttlL /(len(bp)/5)) + " Huffpost NN: " + str(HttlNet/(len(bp)/5)))
		print("F1- NYT SVM: " + str(NttlS /(len(bP)/5)) + " NYT KNN: " + str(NttlK/(len(bp)/5)) + " NYT NB: " + str(NttlN /(len(bp)/5)) + " NYT LC: " +str(NttlL /(len(bp)/5)) + " NYT NN: " + str(NttlNet/(len(bp)/5)))
		
	  
orchestrator = Orchestrator()
<<<<<<< HEAD
splits = orchestrator.read_data(ApplicationConstants.all_articles_random, clean=False, save=False, number_of_articles=1000) 
orchestrator.train_all(splits)
=======
splits = orchestrator.read_data(ApplicationConstants.all_articles_random, clean=True, save=False, number_of_articles=50) 
>>>>>>> 4f622bea4e70761b3ed1c1bb4b2f779a9e6e874f
#cleaned_splits = orchestrator.read_data(ApplicationConstants.cleaned_news_root_path, clean=False, save=False, number_of_articles=1000)

#train embeddings - uncleaned 
#leanings_articles = list(map(lambda leaning: splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test], splits[0]))
#leanings = []

#for leaning in splits[0]:
#	for article in range(len(splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test])):
#		leanings.append(leaning) 

<<<<<<< HEAD
#flat_list = [item for sublist in leanings_articles for item in sublist]

#articles = list(map(lambda article: article.Content, flat_list))  
#labels = list(map(lambda article: article.Label.TargetGender, flat_list))
=======
flat_list = [item for sublist in leanings_articles for item in sublist]
articles = list(map(lambda article: article.Content, flat_list))  
labels = list(map(lambda article: article.Label.TargetGender, flat_list))
>>>>>>> 4f622bea4e70761b3ed1c1bb4b2f779a9e6e874f

#orchestrator.train_sent_models(articles, labels, leanings, ApplicationConstants.all_articles_doc2vec_labels_uncleaned_path, ApplicationConstants.all_articles_doc2vec_vector_uncleaned_path, ApplicationConstants.all_articles_doc2vec_model_uncleaned_path, ApplicationConstants.imdb_sentiment_label_uncleaned_path, ApplicationConstants.imdb_sentiment_vector_uncleaned_path)

#train embeddings - cleaned 
# leanings_articles = list(map(lambda leaning: cleaned_splits[0][leaning][ApplicationConstants.Train] + cleaned_splits[0][leaning][ApplicationConstants.Validation] + cleaned_splits[0][leaning][ApplicationConstants.Test], cleaned_splits[0]))
# leanings = []

# for leaning in cleaned_splits[0]:
# 	for article in range(len(cleaned_splits[0][leaning][ApplicationConstants.Train] + cleaned_splits[0][leaning][ApplicationConstants.Validation] + cleaned_splits[0][leaning][ApplicationConstants.Test])):
# 		leanings.append(leaning) 

# flat_list = [item for sublist in leanings_articles for item in sublist]
# cleaned_articles = list(map(lambda article: article.Content, flat_list))  
# cleaned_labels = list(map(lambda article: article.Label.TargetGender, flat_list))

orchestrator.train_sent_models(articles, labels, leanings, ApplicationConstants.all_articles_doc2vec_labels_cleaned_path, ApplicationConstants.all_articles_doc2vec_vector_cleaned_path, ApplicationConstants.all_articles_doc2vec_model_cleaned_path, ApplicationConstants.imdb_sentiment_label_cleaned_path, ApplicationConstants.imdb_sentiment_vector_cleaned_path)






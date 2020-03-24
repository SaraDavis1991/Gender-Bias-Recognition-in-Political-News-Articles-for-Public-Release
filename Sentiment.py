
from DataReader import DataReader
from Models.SVM_engine import SVM
import numpy as np 
from doc2vec import doc
import os
import ApplicationConstants
from imdb_data import LabeledLineSentence
from Visualizer import Visualizer

class Sentiment():

	def __init__(self):

		self.docEmbed = doc()
		self.Visualizer = Visualizer() 

	def imdb(self, model, label_path, vector_path):
		sources = {'store/test-neg.txt':'TEST_NEG', 'store/test-pos.txt':'TEST_POS', 'store/train-neg.txt':'TRAIN_NEG', 'store/train-pos.txt':'TRAIN_POS' }
		sentences = LabeledLineSentence(sources)
		vectors, labels = sentences.generate_imdb_vec(model, label_path, vector_path)
		return vectors, labels

	def train_sent_models(self, all_articles, leanings, article_doc2vec_label_path, article_doc2vec_vector_path, article_doc2vec_model_path, imdb_label_path, imdb_vector_path):

		articles = list(map(lambda article: article.Content, all_articles))
		labels = list(map(lambda article: article.Label.TargetGender, all_articles))

		if (not os.path.exists(article_doc2vec_model_path)):
			all_articles_model = self.docEmbed.Embed(articles, labels) 
			all_articles_model.save(article_doc2vec_model_path)
		else:
			all_articles_model = self.docEmbed.Load_Model(article_doc2vec_model_path)

		if (not os.path.exists(article_doc2vec_label_path) or not os.path.exists(article_doc2vec_vector_path)):

			all_articles_labels, all_articles_vectors = self.docEmbed.gen_vec(all_articles_model, articles, labels)
			np.save(article_doc2vec_label_path, all_articles_labels)
			np.save(article_doc2vec_vector_path, all_articles_vectors)
			
		else:

			all_articles_labels = np.load(article_doc2vec_label_path)
			all_articles_vectors = np.load(article_doc2vec_vector_path)			

		imdb_vec, imdb_labels = self.imdb(all_articles_model, imdb_label_path, imdb_vector_path)

		models = [SVM()]
		sents = [] 

		for model in models:

			male = []
			female = [] 
			model.Train(imdb_vec, imdb_labels, None, None)
			predictions, confidences = model.Predict(all_articles_vectors, True)
			
			for index, prediction in enumerate(predictions):

				sents.append((all_articles[index], labels[index], prediction, confidences[index], leanings[index]))				  
				if int(labels[index]) == ApplicationConstants.female_value:
					female.append((leanings[index], prediction, confidences[index]))
				else:
					male.append((leanings[index], prediction, confidences[index]))

			self.print_sents(sents)
			self.Visualizer.graph_sentiment(female, male)

	
			bFn, bFp, fFn, fFp, uFn, uFp, hFp, hFn, nFp, nFn = self.calc_plane_dist(female)
			bMn, bMp, fMn, fMp, uMn, uMp, hMp, hMn, nMp, nMn = self.calc_plane_dist(male)
			print("In order female pos/female neg/male pos/male neg")
			print("Breitbart: " + str(bFp) + " " + str(bFn) + " " + str(bMp) + " " + str(bMn))
			print("Fox: " + str(fFp) + " " + str(fFn) + " " + str(fMp) + " " + str(fMn))
			print("USA: " + str(uFp) + " " + str(uFn) + " " + str(uMp) + " " + str(uMn))
			print("Huffpost: " + str(hFp) + " " + str(hFn) + " "+  str(hMp) + " " + str(hMn))
			print("NYT: " + str(nFp) + " " + str(nFn) + " " + str(nMp) + " " + str(nMn))

			self.Visualizer.graph_sentiment(female, male)

	
	def print_sents(self, sents):

		for sent in sents:

			print('\n')

			print('leaning', sent[4])
			print("content:", sent[0].Content)
			print("target:", sent[0].Label.TargetName)
			print("prediction", sent[2])
			print("confidence", sent[3])

if __name__ == "__main__":

	sentiment = Sentiment()
	reader = DataReader()

	splits = reader.Load_Splits(ApplicationConstants.all_articles_random_v2_cleaned, None, number_of_articles=50, clean=False, save=False, shouldRandomize=False)
  
	leanings_articles = list(map(lambda leaning: splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test], splits[0]))
	
	leanings = [] #flattened leanings

	for leaning in splits[0]:
		for article in range(len(splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test])):
			leanings.append(leaning)

	articles = [item for sublist in leanings_articles for item in sublist]
	sentiment.train_sent_models(articles, leanings, ApplicationConstants.all_articles_doc2vec_labels_cleaned_path, ApplicationConstants.all_articles_doc2vec_vector_cleaned_path, ApplicationConstants.all_articles_doc2vec_model_cleaned_path, ApplicationConstants.imdb_sentiment_label_cleaned_path, ApplicationConstants.imdb_sentiment_vector_cleaned_path)
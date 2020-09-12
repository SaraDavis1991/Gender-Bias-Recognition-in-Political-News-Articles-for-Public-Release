#################################
# Bow.py:
# This class is used to generate the most descriptive word list for a given set of empeddings. 
#################################

import os.path
import random
import numpy as np
import pickle
from DataReader import DataReader
import ApplicationConstants
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from Models.SVM_engine import SVM
import nltk

class Bow():

	def calc_count_doc_count_vector(self, word_vector, article):
		words = nltk.word_tokenize(article)
		count_vector = []
		for i in range(len(word_vector)):
			count_vector.append(0)         
		for word in words:
			if '.' in word or ',' in word:
				word = word[:-1]
			word = word.lower()
			if word in word_vector:
				ind = word_vector.index(word)
				count_vector[ind] += 1
			#print(sum(count_vector))
		return count_vector

	def calc_word_vector(self, all_articles):
		word_vector = []
		count_vector = []
		from nltk.corpus import stopwords
		stops = list(stopwords.words('english'))
		punctuation = [',', '.', '\"','"','!', '?', '\'', '$', ''', '\n', ' ', '-', '_', ':', ';', '%', '—', '–', ''', '•']
		no_no = ['oval','Oval', 'Rep', 'rep', 'Rep.', 'rep.', 'Dem.', 'Dem', 'dem', 'dem.', 'son', 'p.m', 'ms']
		articles = list(map(lambda article: article.Content, all_articles))
		for article in articles:
			words = nltk.word_tokenize(article)
			for word in words:
				if '.' in word or ',' in word:
					word = word[:-1]
				word = word.lower()
				if word not in no_no and word not in punctuation and word not in word_vector and word not in stops:
					word_vector.append(word)
		return word_vector

	def load_cumulative_word_vec(self, splits, verbose=False):

		leanings_articles = list(map(lambda leaning: splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test], splits[0]))
		
		#flattened leanings
		leanings = [] 

		#only care about one split
		for leaning in splits[0]:
			for article in range(len(splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test])):
				leanings.append(leaning)

		#grab the articles
		articles = [item for sublist in leanings_articles for item in sublist]

		if os.path.isfile('store/np_cum_vec.npy'):
			numpy_cumulative = np.load('store/np_cum_vec.npy')
			cumulative_word_vec = numpy_cumulative.tolist()
		else:
			cumulative_word_vec = self.calc_word_vector(articles)
			numpy_cumulative = np.array(cumulative_word_vec)
			np.save('store/np_cum_vec.npy', numpy_cumulative)

			if verbose:
				print("total num words = " + str(len(cumulative_word_vec)))

		return cumulative_word_vec, articles

	def generate_bow(self, splits, verbose=True): 

		cumulative_word_vec, articles = self.load_cumulative_word_vec(splits, verbose)

		articles_list = list(map(lambda article: article.Content, articles))
		labels = list(map(lambda article: article.Label.TargetGender, articles))

		if verbose: print("zipping and shuffling")
		zippedArticles = list(zip(articles_list, labels))
		random.shuffle(zippedArticles)
	
		list_articles = []
		list_labels = []

		if verbose: print("unzipping")

		for article, label in zippedArticles:
			list_articles.append(article)
			list_labels.append(label)

		if verbose: print("enumerating")

		for i, label in enumerate(list_labels):
			if label == 0:
				list_labels[i] = -1

		if verbose: print("appending")

		count_vectors = []

		if verbose: print(len(list_articles))

		if os.path.isfile('store/np_count_vec.npy'):
			numpy_cumulative = np.load('store/np_count_vec.npy')
			count_vectors = numpy_cumulative.tolist()
		else:
			for article in list_articles:
				count_vectors.append(self.calc_count_doc_count_vector(cumulative_word_vec, article))
		   
			numpy_count = np.array(count_vectors)
			np.save('store/np_count_vec.npy', numpy_count)

		trainLen = int(len(count_vectors)*0.8)
		
		if verbose: print("building net")

		net = SVM()

		if verbose: print("training")
		net.Train(count_vectors[:trainLen], list_labels[:trainLen], count_vectors[trainLen:], list_labels[trainLen:])
		weights = net.Get_Weights()

		if verbose: print("at preds")
		predictions = net.Predict(count_vectors[trainLen:])
		if verbose:  print(len(predictions), len(list_labels))

		acc = accuracy_score(list_labels[trainLen:], predictions)
		target_names = ['Female', 'Male']

		if verbose: print("accuracy is: " + str(acc))
		if verbose: print(classification_report(list_labels[trainLen:], predictions, target_names=target_names))

		weights = weights[0]

		if verbose: print(weights)

		resTop = sorted(range(len(weights)), key = lambda sub: weights[sub])[-21:]
		resBottom = sorted(range(len(weights)), key = lambda sub: weights[sub])[:21]

		pickle.dump(net, open("store/perceptron2.sav", 'wb'))

		return cumulative_word_vec, weights, resTop, resBottom

	def print_bow(self, cumulative_word_vec, weights, resTop, resBottom):

		print("Male Top Words: ")
		for index in resTop:
			print(cumulative_word_vec[index], float(weights[index]))

		print("Female Top Words: ")
		for index in resBottom:
			print(cumulative_word_vec[index], float(weights[index]))

if __name__ == "__main__":

	bow = Bow()
	reader = DataReader()  

	splits = reader.Load_Splits(ApplicationConstants.all_articles_random_v2_cleaned, None, number_of_articles=50, clean=False, save=False, shouldRandomize=False)

	cumulative_word_vec, weights, resTop, resBottom = bow.generate_bow(splits)
	bow.print_bow(cumulative_word_vec, weights, resTop, resBottom)


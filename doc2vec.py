from __future__ import print_function, division
from matplotlib import pyplot as plt
import json
import numpy as np

import ApplicationConstants
from DataReader import DataReader
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
from nltk.tokenize import word_tokenize
import random 
import os.path

class doc():

	# trains and returns the vector embeddings for doc2vec or sent2vec
	#	Parameters:5
	#	articles: a list of articles that are cleaned
	#	labels: a list of labels corresponding to the article genders
	def embed_fold(self, articles, labels, leaning):

		model = self.Embed(articles, labels)
		targets, regressors = self.gen_vec(model, articles, labels)

		return list(targets), regressors, model

	def Load_Model(self, article_doc2vec_model_path):
		if (os.path.exists(article_doc2vec_model_path)):
			model = Doc2Vec.load(article_doc2vec_model_path)
			return model 
		return None

	def Embed(self, articles, labels, vector_size=50, epochs=100):

		tagged_doc_articles = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[labels[i]]) for i, _d in enumerate(articles)]
		#random.shuffle(tagged_doc_articles)

		#dm 1 is pv-dm, dm 0 is pv-dbow size is feature vec size, alpha is lr, negative is noise words, sample is thresh for down smample
		model = Doc2Vec(vector_size=vector_size, alpha = 0.001, min_alpha = 0.00025, min_count = 1, epochs=epochs, negative=1, dm = 0, workers = multiprocessing.cpu_count()) 
		model.build_vocab(tagged_doc_articles)
		model.train(tagged_doc_articles, total_examples = model.corpus_count, epochs= model.epochs)

		return model

		#model.save("d2v.model")
		#print("model saved")

	def fine_tune(self, articles, labels, model, epochs=50, learning_rate=0.0001): 

		tagged_doc_articles = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[labels[i]]) for i, _d in enumerate(articles)]
		model.train(tagged_doc_articles, total_examples = model.corpus_count, epochs=epochs, start_alpha=learning_rate, end_alpha=learning_rate)

		return model 
	
	def gen_vec(self, model, articles, labels):
		tagged_doc_articles = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[labels[i]]) for i, _d in enumerate(articles)]
		sents = tagged_doc_articles
		targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
		return targets, feature_vectors





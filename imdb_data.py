# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys


from Models.SVM_engine import SVM
from Models.KNN_engine import KNN
from Models.Naive_Bayes_engine import Naive_Bayes
from Models.Linear_Classification_engine import Linear_Classifier 
from Models.NN_engine import NN



class imdb(object):
	def __init__(self):
		sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}
		self.sources = sources

		flipped = {}

		# make sure that keys are unique
		for key, value in sources.items():
			if value not in flipped:
				flipped[value] = [key]
			else:
				raise Exception('Non-unique prefix encountered')

	def __iter__(self):
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

	def to_array(self):
		self.sentences = []
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					self.sentences.append(LabeledSentence(
						utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
		return self.sentences

	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences

	def generate_imdb_vec(self, sentences):
		model = Doc2Vec.load("all_1.model")
		imdbVectors = model.infer_vector(sentences)
		return imdbVectors

	
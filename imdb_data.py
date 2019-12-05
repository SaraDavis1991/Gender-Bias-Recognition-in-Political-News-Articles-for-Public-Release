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
from smart_open import open
from Models.SVM_engine import SVM
from Models.KNN_engine import KNN
from Models.Naive_Bayes_engine import Naive_Bayes
from Models.Linear_Classification_engine import Linear_Classifier 
from Models.NN_engine import NN

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
import _pickle as pickle
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):
	def __init__(self, sources):
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
			with open(source) as fin:
				for item_no, line in enumerate(fin):
					self.sentences.append(LabeledSentence(
						utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
		return self.sentences
	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences

	def train(self):
		model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
		model.build_vocab(self.to_array())
		for epoch in range(50):
			logger.info('Epoch %d' % epoch)
			model.train(self.sentences_perm(),
						total_examples=model.corpus_count,
						epochs=model.iter,
			)
		model.save('./imdb.d2v')

	def generate_imdb_vec(self):
		model = Doc2Vec.load("all_1.model")
		imdbVectors = model.infer_vector(self.sources)
		return imdbVectors

	
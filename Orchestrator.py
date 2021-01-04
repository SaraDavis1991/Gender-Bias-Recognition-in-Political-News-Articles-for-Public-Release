#################################
# Orchestrator.py:
# Runs various processes that are shared by numerous classes.
#################################

from DataReader import DataReader
from DataContracts import Article
from doc2vec import doc
from Metrics import Metrics
from Visualizer import Visualizer
from imdb_data import LabeledLineSentence
import ApplicationConstants
import StopWords
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from Models.SVM_engine import SVM
from Models.KNN_engine import KNN
from Models.Naive_Bayes_engine import Naive_Bayes
from Models.Linear_Classification_engine import Linear_Classifier
from Models.NN_engine import NN
from Models.NN_engine import  Linear_NN
import statistics
import numpy as np 
import matplotlib.pyplot as plt
import os.path
import timeit
import spacy
import nltk
import re
import random
import pickle
import ngrams as ngram

class Orchestrator():

	def __init__(self):
		self.Reader = DataReader()
		
		self.Splits = None 
		self.Sources = None
		self.docEmbed = doc()
		self.Metrics = Metrics()
		self.Visualizer = Visualizer()

	def read_data(self, path, savePath=None, clean=True, save=False, random=False, number_of_articles = 50, pos_tagged = False):
		return self.Reader.Load_Splits(path, savePath=savePath, clean=clean, save=save, shouldRandomize=random, number_of_articles=number_of_articles, pos_tagged = pos_tagged)

	def read_data_csv(self, path, savePath=None, clean=True, save=False, random=False, number_of_articles = 50):
		return self.Reader.Load_ATN_csv(path, savePath=savePath, clean=clean, save=save, shouldRandomize=random, number_of_articles=number_of_articles)

	def imdb(self, model, label_path, vector_path):
		sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS' }
		sentences = LabeledLineSentence(sources)
		vectors, labels = sentences.generate_imdb_vec(model, label_path, vector_path)
		return vectors, labels

	def print_sents(self, sents):

		for sent in sents:

			print('\n')

			print('leaning', sent[4])
			print("content:", sent[0].Content)
			print("target:", sent[0].Label.TargetName)
			print("prediction", sent[2])
			print("confidence", sent[3])

	def calc_plane_dist(self, gender):
		ttlBreitbartPos,ttlBreitbartNeg, breitbartneg, breitbartpos, ttlFoxPos, ttlFoxNeg, foxneg, foxpos, ttlUsaPos, ttlUsaNeg, usaneg, usapos, ttlHuffPos, ttlHufNeg, huffneg, huffpos, ttlNytPos, ttlNytNeg, nytneg, nytpos = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

		for i in range(len(gender)):
			if gender[i][0] == 'breitbart':
			
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlBreitbartNeg +=1
						breitbartneg += gender[i][2]
					else:
						ttlBreitbartPos +=1
						breitbartpos += gender[i][2]
			if gender[i][0] == 'fox' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlFoxNeg+=1
						foxneg += gender[i][2]
					else:
						ttlFoxPos +=1
						foxpos += gender[i][2]
			if gender[i][0] == 'usa_today' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlUsaNeg +=1
						usaneg += gender[i][2]
					else:
						ttlUsaPos +=1
						usapos += gender[i][2]
			if gender[i][0] == 'huffpost' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlHufNeg+=1
						huffneg += gender[i][2]
					else:
						ttlHuffPos +=1
						huffpos += gender[i][2]
			if gender[i][0] == 'new_york_times' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlNytNeg +=1
						nytneg += gender[i][2]
					else:
						ttlNytPos +=1
						nytpos += gender[i][2]

		if breitbartneg != 0:
			breitbartneg = breitbartneg/ttlBreitbartNeg
		else:
			breitbartneg = 0
		if breitbartpos != 0:
			breitbartpos = breitbartpos/ttlBreitbartPos
		else:
			breitbartpos = 0
		if foxneg != 0:
			foxneg = foxneg/ttlFoxNeg
		else:
			foxneg = 0
		if foxpos != 0:
			foxpos = foxpos /ttlFoxPos
		else:
			foxpos = 0
		if usaneg != 0:
			usaneg = usaneg/ttlUsaNeg
		else:
			usaneg = 0
		if usapos != 0:
			usapos = usapos /ttlUsaPos
		else:
			usapos = 0
		if huffneg != 0:
			huffneg = huffneg/ttlHufNeg
		else:
			huffneg = 0
		if huffpos != 0:
			huffpos = huffpos /ttlHuffPos
		else:
			huffpos = 0
		if nytneg != 0:
			nytneg = nytneg/ttlNytNeg
		else:
			nytneg = 0
		if nytpos != 0:
			nytpos = nytpos /ttlNytPos
		else:
			np = 0
		
		return breitbartneg, breitbartpos, foxneg, foxpos, usaneg, usapos, huffneg, huffpos, nytneg, nytpos

	def print(self, fileName, allF, allM):
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

		model = self.docEmbed.Embed(articles, labels)
		targets, regressors = self.docEmbed.gen_vec(model, articles, labels)

		return list(targets), regressors, model
	
	def train_all(self, splits):
		''' trains all models against all leanings
		
		Parameters: 
		------------
		splits: A list of the splits 

		''' 
		#models = [SVM(), KNN(), Naive_Bayes(), Linear_Classifier(), NN()]
		models = [NN()]
		bP, bR, bF, fP, fR, fF, uP, uR, uF, hP, hR, hF, nP, nR, nF = [], [], [], [], [], [], [], [], [], [], [], [], [], []



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

				

				if split_count == 1:
					self.Visualizer.plot_TSNE(leaning, training_embeddings + validation_embeddings + test_embeddings, training_labels + validation_labels + test_labels, training_dataset + validation_dataset + test_dataset)
		

	def get_most_sig_sent(self, articles, context_sentence_number = 2):

		for article_index, article in enumerate(articles): 

			target = article.Label.TargetName
			lastname = target.split('_')[1]
			qualtive_sentences = []
			sentences = article.Content.split('.')

			#start 2 over so we have context
			for sentence_index in range(2, len(sentences)):

				#check if name in sentence
				if lastname in sentences[sentence_index]:
					
					for context_number in range(-context_sentence_number, context_sentence_number):

						if context_number + sentence_index < len(sentences):
							qualtive_sentences.append(sentences[sentence_index + context_number])

					break 

			articles[article_index].Content = " ".join(qualtive_sentences)

		return articles

	def calc_metrics(self, bP, fP, uP, hP, nP ):
		BreitbartTtlSvm, BreitbartTtlKnn, BreitbartTtlNB, BreitbartTtlLin, BreitbartTtlNN, FoxTtlSvm, FoxTtlKnn, FoxTtlNB, FoxTtlLin, FoxTtlNN, UsaTtlSvm, UsaTtlKnn, UsaTtlNB, UsaTtlLin, UsaTtlNN  = 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0
		HuffTtlSvm, HuffTtlKnn, HuffTtlNB, HuffTtlLin, HuffTtlNN, NytTtlSvm, NytTtlKnn, NytTtlNB, NytTtlLin, NytTtlNN = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


		for i in range(len(bP)):
			if i %5 == 0:
				BreitbartTtlSvm +=bP[i]
				FoxTtlSvm +=fP[i]
				UsaTtlSvm += uP[i]
				HuffTtlSvm += hP[i]
				NytTtlSvm += nP[i]
			if i % 5 == 1:
				BreitbartTtlKnn +=bP[i]
				FoxTtlKnn +=fP[i]
				UsaTtlKnn += uP[i]
				HuffTtlKnn += hP[i]
				NytTtlKnn += nP[i]
			if i % 5 == 2:
				BreitbartTtlNB +=bP[i]
				FoxTtlNB +=fP[i]
				UsaTtlNB += uP[i]
				HuffTtlNB += hP[i]
				NytTtlNB += nP[i]
			if i %5 == 3:
				BreitbartTtlLin +=bP[i]
				FoxTtlLin +=fP[i]
				UsaTtlLin += uP[i]
				HuffTtlLin += hP[i]
				NytTtlLin += nP[i]
			if i%5 == 4:
				BreitbartTtlNN +=bP[i]
				FoxTtlNN +=fP[i]
				UsaTtlNN += uP[i]
				HuffTtlNN += hP[i]
				NytTtlNN += nP[i]
		bp = bP
		print("Breitbart SVM: " + str(BreitbartTtlSvm /(len(bP)/5)) + " Breitbart KNN: " + str(BreitbartTtlKnn/(len(bP)/5)) + " Breitbart NB: " + str(BreitbartTtlNB /(len(bP)/5)) + " Breitbart LC: " +str(BreitbartTtlLin /(len(bP)/5)) + " Breitbart NN: " + str(BreitbartTtlNN/(len(bP)/5)))
		print("Fox SVM: " + str(FoxTtlSvm /(len(bP)/5)) + " Fox KNN: " + str(FoxTtlKnn/(len(bP)/5)) + " Fox NB: " + str(FoxTtlNB /(len(bP)/5)) + " Fox LC: " +str(FoxTtlLin /(len(bP)/5)) + " Fox NN: " + str(FoxTtlNN/(len(bP)/5)))
		print("USA SVM: " + str(UsaTtlSvm/(len(bP)/5)) + " USA KNN: " + str(UsaTtlKnn/(len(bP)/5)) + " USA NB: " + str(UsaTtlNB /(len(bP)/5)) + " USA LC: " +str(UsaTtlLin /(len(bP)/5)) + " USA NN: " + str(UsaTtlNN/(len(bP)/5)))
		print("Huffpost SVM: " + str(HuffTtlSvm /(len(bP)/5)) + " Huffpost KNN: " + str(HuffTtlKnn/(len(bp)/5)) + " Huffpost NB: " + str(HuffTtlNB /(len(bp)/5)) + " Huffpost LC: " +str(HuffTtlLin /(len(bp)/5)) + " Huffpost NN: " + str(HuffTtlNN/(len(bp)/5)))
		print("NYT SVM: " + str(NytTtlSvm /(len(bP)/5)) + " NYT KNN: " + str(NytTtlKnn/(len(bp)/5)) + " NYT NB: " + str(NytTtlNB /(len(bp)/5)) + " NYT LC: " +str(NytTtlLin /(len(bp)/5)) + " NYT NN: " + str(NytTtlNN/(len(bp)/5)))

	def check_word_content(self, word_list, all_articles):
		articles = list(map(lambda article: article.Content, all_articles))

		bad_words = []
		ttl = 0
		#count = 0
		affected = []
		count_list = []
		print(len(articles))
		for i, article in enumerate(articles):
			for word in word_list:
				if re.search(rf'\b{word}\b' ,  article):
					if word not in bad_words:
						bad_words.append(word)
						count_list.append(1)

					else:
						ind = bad_words.index(word)
						val = count_list[ind]

						num_exist = val + article.count(word)
						count_list[ind] = num_exist

					affected.append(i)

		a = []
		print(affected)
		[a.append(x) for x in affected if x not in a]
		print(a)
		print(bad_words)
		print("total articles: ", len(a))
		zipped = list(zip(bad_words, count_list))
		print("total use: ",zipped)

	def construct_vocabulary(self, ngram_dict, ngrams, splittype):
		for ngram in ngrams:
			# print(ngram)
			if ngram in ngram_dict.keys():
				ngram_dict[ngram] += 1
			elif splittype == "train":
				ngram_dict[ngram] = 1
			elif "UNK" in ngram_dict.keys() and splittype == "test":
				ngram_dict["UNK"] += 1
			else:
				ngram_dict["UNK"] = 1



	# print(vocabulary)

	def winnow_vocab(self, ngram_dict):
		# sorted_dict = dict(sorted(ngram_dict.items(), key = lambda item : item[1], reverse = True))
		# keys = list(sorted_dict.keys())[:300000]
		# vocab_counter = dict.fromkeys(keys, 0)
		vocab = {}

		for word, count in ngram_dict.items():
			if count > 1:
				vocab[word] = count
		return vocab

	def build_bow(self, vocab_counter, article):
		# print(len(vocab))
		# vocab_counter = dict.fromkeys(vocab, 0)
		# for i in range(len(vocab)):
		#    vocab_counter.append(0)

		for word in article:
			# ind = vocab.index(word)
			# vocab_counter[ind] +=1
			if word not in vocab_counter:
				word = 'UNK'
			vocab_counter[word] += 1

	def construct_counts(self,type, n, vocab_counter, corpus, savetype):
		counted_set = []
		keys = vocab_counter.keys()

		for i, text in enumerate(corpus):
			vocab_counter = dict.fromkeys(keys, 0)
			self.build_bow(vocab_counter, text)
			#print(len(vocab_counter.values()))
			lencounts = len(vocab_counter.values())
			values =np.asarray(list(vocab_counter.values()))
			#print(np.shape(values))
			counted_set.append(values) #make an array out of vocab_counter
		# print(i)
		counted_set = np.asarray(counted_set) #make an array out of counted set
		print("COUNTED SET SHAPE, SHOULD BE NUM ARTICLES * VOCAB", np.shape(counted_set))
		corpus_numpy_save = "./store/" + type + "_" + str(n) + "gram_CorpusCounts" + savetype + ".npy"
		np.save(corpus_numpy_save, counted_set) #save the array
	def run_vocab_construction(self , vocabulary):

		if len(vocabulary) > 300000:
			print("winnnowing")
			char_vocab = self.winnow_vocab(vocabulary)
			char_vocab['UNK'] = 1
		else:
			char_vocab = vocabulary
			char_vocab['UNK'] = 1


		print("VOCAB", len(char_vocab))
		return char_vocab
	def do_trainTestValSplit(self,i,ngramtype, gram_vocab, corpus, all_articles, splittype, pos):
		for filecontents in all_articles:
			filecontent_word_gram = ngram.doc_word_ngram(i, filecontents)
			#gram_vocab = self.run_vocab_construction(ngramtype, i, gram_vocab, filecontent_word_gram)
			self.construct_vocabulary(gram_vocab, filecontent_word_gram, splittype) #construct ngrams
			corpus.append(filecontent_word_gram) #append to corpus
		print("NUM ARTICLES ", splittype,  len(corpus))
		if splittype == "train" or splittype == "":
			gram_vocab = self.run_vocab_construction(gram_vocab) #get winnowed vocab
		print("VOCAB LEN", len(gram_vocab))
		if "UNK" not in gram_vocab.keys():
			gram_vocab['UNK'] = 1
		self.construct_counts(ngramtype, i, gram_vocab, corpus, splittype) #turn words into counts
		if pos:
			type = "pos"
		else:
			type = "word"
		if splittype == "train" or "":
			vocab_numpy_save = "./store/" + type + "_" + str(i) + "gram_vocab.npy"
			words = np.asarray(list(gram_vocab.keys()))
			np.save(vocab_numpy_save, words)
		del corpus
		return gram_vocab

	def calc_ngram_vectors(self, all_articles_const, pos):
		#get all articles
		all_articles_train = []
		all_articles_test = []
		all_articles_validation = []
		#for i, split in enumerate(all_articles_const):
			#print("Fold " + str(i + 1))
		for j, leaning in enumerate(all_articles_const[0]):
			print("calc word vec", leaning)
			training_dataset = all_articles_const[0][leaning][ApplicationConstants.Train]
			validation_dataset = all_articles_const[0][leaning][ApplicationConstants.Validation]
			test_dataset = all_articles_const[0][leaning][ApplicationConstants.Test]
			#all_articles = list(map(lambda art: art.Content, training_dataset + validation_dataset + test_dataset))
			all_articles_train.append(list(map(lambda art: art.Content, training_dataset)))
			all_articles_test.append(list(map(lambda art: art.Content, test_dataset)))
			all_articles_validation.append(list(map(lambda art: art.Content,  validation_dataset)))
		all_articles_train = [j for sub in all_articles_train for j in sub]
		all_articles_test = [j for sub in all_articles_test for j in sub]
		all_articles_validation = [j for sub in all_articles_validation for j in sub]
		all_articles = all_articles_train+all_articles_validation
		print("TRAIN + VAL", len(all_articles))
		print("TEST", len(all_articles_test))
		if not pos:
			#do char grams
			for i in range(2, 6):
				#for filecontents in all_articles:
					#filecontent_char_gram = ngram.doc_char_ngram(i, filecontents)
					#char_gram_vocab = self.run_vocab_construction("char", i,  char_gram_vocab, filecontent_char_gram)
					#char_gram_corpus.append(filecontent_char_gram)
					#char_corpus_counts = self.construct_counts("char",i, char_gram_vocab, char_gram_corpus)
				#del char_gram_corpus
				#del char_gram_vocab
				'''
				for filecontents in all_articles_train:
					filecontent_word_gram = ngram.doc_word_ngram(i, filecontents)
					word_gram_vocab = self.run_vocab_construction("word",i,  word_gram_vocab, filecontent_word_gram)
					word_gram_corpus.append(filecontent_word_gram)
				self.construct_counts("word", i, word_gram_vocab, word_gram_corpus, "train")
				if pos:
					type = "pos"
				else:
					type = "word"
				vocab_numpy_save = "./store/" + type + "_" + str(i) + "gram_vocab_train.pkl"
				f = open(vocab_numpy_save, "wb")
				pickle.dump(word_gram_vocab, f)
				f.close()
				del word_gram_corpus
				del word_gram_vocab
				'''
				#self.do_trainTestValSplit( i, "word", word_gram_vocab, word_gram_corpus, all_articles_train, "train", pos)
				#self.do_trainTestValSplit(i, "word", word_gram_vocab, word_gram_corpus, all_articles_validation, "validation", pos)
				#self.do_trainTestValSplit(i, "word", word_gram_vocab, word_gram_corpus, all_articles_test, "test", pos)
				word_gram_vocab = {}
				word_gram_corpus = []
				word_gram_vocab = self.do_trainTestValSplit(i, "word", word_gram_vocab, word_gram_corpus, all_articles, "train", pos)
				word_gram_corpus = []
				word_gram_vocab = dict.fromkeys(word_gram_vocab, 0) #reset values of dictionary to 0 but maintain words
				self.do_trainTestValSplit(i, "word", word_gram_vocab, word_gram_corpus, all_articles_test, "test", pos)

		else:
			for i in range(2, 6):

				'''
				for filecontents in all_articles:
					filecontent_pos_gram = ngram.doc_pos_ngram(i, filecontents)
					pos_gram_vocab = self.run_vocab_construction("pos", i, pos_gram_vocab, filecontent_pos_gram)
					pos_gram_corpus.append(filecontent_pos_gram)
				self.construct_counts("pos",i,  pos_gram_vocab, pos_gram_corpus)
				vocab_numpy_save = "./store/" + type + "_" + str(i) + "gram_vocab.pkl"
				f = open(vocab_numpy_save, "wb")
				pickle.dump(pos_gram_vocab, f)
				f.close()
				del pos_gram_corpus
				del pos_corpus_counts
				'''
				#self.do_trainTestValSplit( i, "pos", pos_gram_vocab, pos_gram_corpus, all_articles_train, "train", pos)
				#self.do_trainTestValSplit(i, "pos", pos_gram_vocab, pos_gram_corpus, all_articles_validation, "validation", pos)
				#self.do_trainTestValSplit(i, "pos", pos_gram_vocab, pos_gram_corpus, all_articles_test, "test", pos)
				pos_gram_vocab = {}
				pos_gram_corpus = []
				self.do_trainTestValSplit(i, "pos", pos_gram_vocab, pos_gram_corpus, all_articles, "train", pos)
				pos_gram_corpus = []
				pos_gram_vocab = dict.fromkeys(pos_gram_vocab, 0) #reset values of dictionary to 0 but maintain words
				self.do_trainTestValSplit(i, "pos", pos_gram_vocab, pos_gram_corpus, all_articles_test, "test", pos)


	def calc_word_vector(self, all_articles, not_pos = True, lemmad = True, print_vocab=False):
		nlp = spacy.load("en_core_web_lg")
		word_vector = set([])
		punctuation = [',', '.', '\"', '"', '!', '?', '\'', '$', ''', '\n', '_', ':', ';', '%', '—', '–', ''','~','―','′', ',', '≠', '|',
					   '•', ' ', ', ', '/', '>', '<', '=', '-', '’', ']', '[', '(', ')', '{', '}', '@', '#', '^', '*', '&', '­']
		if not_pos:
			from nltk.corpus import stopwords
			stops = list(stopwords.words('english'))
		for i, split in enumerate(all_articles):
			print("Fold " + str(i + 1))
			for j, leaning in enumerate(split):
				print("calc word vec", leaning)
				training_dataset = split[leaning][ApplicationConstants.Train]
				validation_dataset = split[leaning][ApplicationConstants.Validation]
				test_dataset = split[leaning][ApplicationConstants.Test]
				all_articles = list(map(lambda art: art.Content, training_dataset + validation_dataset + test_dataset))
				print("num articles to calc word vec", str(len(all_articles)))
				for article in all_articles:
					document = nlp(article)
					if not_pos:
						for token in document:
							if not token.is_punct:
								if not lemmad:
									word = token.orth_.lower()
								else:
									word = token.lemma_.lower()

								if len(word) > 1:
									if word[0] in punctuation:
										word = word[1:]
									if word[-1] in punctuation:
										word = word[:-1]
								for punct in punctuation:
									if punct in word:
										word = '.'
								if "gpe" in word:
									word = "gpe"
								if "norp" in word:
									word = "norp"
								replacement_words = ["norp", "gpe", "loc", "person", "people"]
								if word not in punctuation and word not in word_vector and word not in stops and word:
									if len(word) >= 2 and word != "\n" and ":" not in word:
										word_vector.add(word)

					else:
						for token in document:
							if token.pos_ is "ADJ" and token.orth_.lower() not in word_vector and token.text not in punctuation:
								if not lemmad:
									word = token.orth_.lower()
								else:
									word = token.lemma_.lower()

								if len(word) > 1:
									if word[0] in punctuation:
										word = word[1:]

									if word[-1] in punctuation:
										word = word[:-1]

								for punct in punctuation:
									if punct in word:
										word = '.'
								if "gpe" in word and "gpe" not in word_vector:
									word = "gpe"

								elif "gpe" in word:
									break
								if "norp" in word and "norp" not in word_vector:
									word = "norp"
								elif "norp" in word:
									break
								if len(word) >= 2 and word != "\n" and ":" not in word:
									word_vector.add(word)


			if os.path.exists("./vocabulary/") == False:
				os.mkdir("./vocabulary/")

			if print_vocab:
				printed_word_vec = sorted(word_vector)
				if not_pos and lemmad:
					name = "vocabulary/fullVocab_lemmad.txt"
				elif not_pos and not lemmad:
					name = "vocabulary/fullVocab_notLemmad.txt"
				elif not not_pos and lemmad:
					name = "vocabulary/adjVocab_lemmad.txt"
				else:
					name = "vocabulary/adjVocab_notLemmad.txt"
				fout = open(name, 'w')
				for item in printed_word_vec:
					fout.write(item + '\n')
				fout.close()

			return list(word_vector)

	def calc_count_doc_count_vector(self, word_vector, article, nlp, lemmad = False):

		words = nlp(article)
		count_vector = [0]*len(word_vector)
		count_set = {}
		punctuation = [',', '.', '\"', '"', '!', '?', '\'', '$', ''', '\n', '_', ':', ';', '%', '—', '–', ''',
					   '•', ' ', ', ', '/', '>', '<', '=', '-', '’', ']', '[', '(', ')', '{', '}', '@', '#', '^', '*',
					   '&', ':']
		for word in word_vector:
			count_set[word] = 0

		for token in words:
			if lemmad:
				word = token.lemma_.lower()
			else:
				word = token.orth_.lower()

			for i, char in enumerate(word):
				if len(word) > 1:
					if word[0] in punctuation:
						word = word[1:]

					if word[-1] in punctuation:
						word = word[:-1]

			for punct in punctuation:
				if punct in word:
					word = 'aklfjakldfjlaskf' #if there's punctuation still in the middle of the word, it's a garbage word, and we insert a garbage word that doesn't exist in the cum_vec
			if "gpe" in word:
				word = "gpe"
			if "norp" in word:
				word = "norp"
			if word in count_set:
				count_set[word] +=1

		for i in range(len(word_vector)):
			count_vector[i] = count_set[word_vector[i]]
		return count_vector

	def get_all_articles(self):
		splits = self.read_data(ApplicationConstants.all_articles_random_v2_cleaned, clean=True, save=True,
										savePath="./Data/articles_random_v2_cleaned.json",
										number_of_articles=50)  # article objects

		leanings_articles = list(map(
			lambda leaning: splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][
				ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test], splits[0]))
		# print(leanings_articles)
		leanings = []  # flattened leanings

		for leaning in splits[0]:
			for article in range(len(splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][
				ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test])):
				leanings.append(leaning)
		articles = [item for sublist in leanings_articles for item in sublist]
		return articles

	def load_exisiting_bow(self, model, file_name_2):
		net = pickle.load(open(model, 'rb'))
		weights = net.Get_Weights()

		numpy_cumulative = np.load(file_name_2)
		count_vectors = numpy_cumulative.tolist()
		label_name = file_name_2[-4] + "_labels.npy"
		list_labels = np.load(label_name)
		list_labels = list_labels.tolist()
		trainLen = int(len(count_vectors) * 0.8)

		predictions = net.Predict(count_vectors[trainLen:])

		acc = accuracy_score(list_labels[trainLen:], predictions)
		target_names = ['Female', 'Male']
		print("accuracy is: " + str(acc))


	def run_bow(self, file_name_1, file_name_2, model_name, do_ngrams = False, not_pos = True, lemmad = True, print_vocab = False, balanced = True):
		label_name = file_name_2[:-4] + "_labels.npy"
		if os.path.exists("./vocabulary/") == False:
			os.mkdir("./vocabulary/")
		if balanced:
			numArticles = 50
		else:
			numArticles = 1000
		if os.path.exists("./BOW_models/") == False:
			os.mkdir("./BOW_models/")
		trainLen = 0
		#if file_name_2 exists, then all np arrays exists. load them and do BOW
		if os.path.isfile(file_name_2) and not do_ngrams:
			numpy_counts = np.load(file_name_2)
			numpy_cum_labels = np.load(label_name)
			count_vectors = numpy_counts.tolist()
			labels = numpy_cum_labels.tolist() #was list_labels
			numpy_cumulative = np.load(file_name_1)
			cumulative_word_vec = numpy_cumulative.tolist()
			#approximate train + val length using .8 because we already have files and can't get exact without reloading all data
			#which takes a long time
			trainLen = int(len(labels) * .8)
		else:
			if os.path.isfile(file_name_1) and not do_ngrams : #check if file_name 1 exists, and load if it does
				numpy_cumulative = np.load(file_name_1)
				cumulative_word_vec = numpy_cumulative.tolist()

			#else: #otherwise, load the correct json
			elif not_pos:
				pos = False
				articles = self.read_data(path=ApplicationConstants.all_articles_random_v4_cleaned, number_of_articles=numArticles
										  ,save=False)
			else:
				pos = True
				articles = self.read_data(path = ApplicationConstants.all_articles_random_v4_cleaned_pos_candidate_names,
										  number_of_articles =numArticles, save = False)

			#create the cumulative word vec for all articles, and save it as numpy array in store directory
			if os.path.isfile(file_name_1) == False and not do_ngrams:
				cumulative_word_vec = self.calc_word_vector(articles, not_pos, lemmad, print_vocab)

			else:
				cumulative_word_vec = self.calc_ngram_vectors(articles, pos)
			numpy_cumulative = np.array(cumulative_word_vec)
			if do_ngrams == False:
				np.save(file_name_1, numpy_cumulative)
				print("store/total num words = " + str(len(cumulative_word_vec)))

			#get all articles in a list
			list_articles_list_train = []
			list_articles_list_val = []
			list_articles_list_test = []
			list_labels_train = []
			list_labels_test = []
			list_labels_val = []

			#need to do for each leaning to get all of the articles from each leaning
			#currently only doing it using first fold split
			for j, leaning in enumerate(articles[0]):

				training_dataset = articles[0][leaning][ApplicationConstants.Train] #load all train for fold
				validation_dataset = articles[0][leaning][ApplicationConstants.Validation] #load all val for fold
				test_dataset = articles[0][leaning][ApplicationConstants.Test] #load all test for fold

				train_articles = list(map(lambda article: article.Content, training_dataset))
				test_articles = list(map(lambda article: article.Content, test_dataset))
				validation_articles = list(map(lambda article: article.Content, validation_dataset))

				#append the articles for the leaning to a master list
				list_articles_list_train.append(train_articles)
				list_articles_list_val.append(validation_articles)
				list_articles_list_test.append(test_articles)

				train_labels = list(map(lambda article: article.Label.TargetGender, training_dataset))
				test_labels = list(map(lambda article: article.Label.TargetGender, test_dataset ))
				validation_labels = list(map(lambda article: article.Label.TargetGender, validation_dataset))

				#append the labels for the leaning to a master list
				list_labels_train.append(train_labels)
				list_labels_test.append(test_labels)
				list_labels_val.append(validation_labels)

			#convert 2d list into 1d
			train_articles = [j for sub in list_articles_list_train for j in sub]

			validation_articles = [j for sub in list_articles_list_val for j in sub]
			test_articles = [j for sub in list_articles_list_test for j in sub]
			train_labels = [j for sub in list_labels_train for j in sub]

			validation_labels = [j for sub in list_labels_val for j in sub]
			test_labels = [j for sub in list_labels_test for j in sub]

			#combine all articles and all labels into one list
			articles_list = train_articles + validation_articles + test_articles
			labels = train_labels + validation_labels + test_labels

			#change the 0 labels to -1 for easier training
			print("enumerating")
			for i, label in enumerate(labels): #was list_labels
				if label == 0:
					labels[i] = -1 #was list_labels

			#Create a word count vector for every article in the dataset and save the count vector in numpy array
			if do_ngrams == False:
				print("appending")
				count_vectors = []
				nlp = spacy.load("en_core_web_lg")
				for article in articles_list:
					count_vectors.append(self.calc_count_doc_count_vector(cumulative_word_vec, article, nlp, lemmad))

				numpy_count = np.array(count_vectors)
				numpy_label = np.array(labels) #was list_labels
				np.save(file_name_2, numpy_count)
				np.save(label_name, numpy_label)
		if do_ngrams ==False:
			loop = 2
		else:
			loop = 6

		#Build and train an SVM BOW
		if trainLen == 0:
			trainLen = len(train_articles) + len(validation_articles) #train on train and val since we're not tuning hyperparams
		print("TRAIN LEN:", str(trainLen))
		acc = 0
		print("building net")
		for l in range(2, loop):
			multiple = False
			print(do_ngrams)
			if do_ngrams == True:
				multiple = True
				if pos:
					type = "pos"
				else:
					type = "word"
				count_vectors_train = np.load("./store/" + type + "_" + str(l) + "gram_CorpusCountstrain.npy", allow_pickle = True) #load the array
				#count_vectors_validation = np.load("./store/" + type + "_" + str(l) + "gram_CorpusCountsvalidation.npy", allow_pickle = True)
				count_vectors_test = np.load("./store/" + type + "_" + str(l) + "gram_CorpusCountstest.npy",allow_pickle=True)
				#count_vectors = count_vectors_train + count_vectors_validation + count_vectors_test
				#count_vectors_train = np.load("./store/" + type + "_" + str(l) + "gram_CorpusCounts.npy", allow_pickle = True)

				word_model ="./store/" + type + "_"+ str(l) +"gram_vocab.npy"
				print(os.path.exists(word_model))
				cumulative_word_vec = np.load(word_model)
				#print("LEN", len(count_vectors_train))
				foutval = "vocabulary/output_words_50Articles_" + type + "_" + str(l) + "gram_lesswinnow.txt"
				fout = open(foutval, 'w')
			print(np.shape(count_vectors_train), np.shape(count_vectors_test))
			net = SVM()
			print("training")
			if not do_ngrams:
				net.Train(count_vectors[:trainLen], labels[:trainLen], count_vectors[:trainLen], labels[:trainLen]) #no validation occurs here, so last 2 params do nothing

			else:
				net.Train(count_vectors_train,train_labels+validation_labels, count_vectors_train, train_labels+validation_labels)
			weights = net.Get_Weights()
			if not do_ngrams:
				predictions = net.Predict(count_vectors[trainLen:]) #pred on test counts
				acc = accuracy_score(labels[trainLen:], predictions)  # get accuracy
				class_rep = classification_report(labels[trainLen:], predictions,
												  target_names=target_names)  # was list_labels
			else:
				predictions = net.Predict(count_vectors_test)
				acc = accuracy_score(test_labels, predictions)
				class_rep = classification_report(test_labels, predictions)


			target_names = ['Female', 'Male']
			print("accuracy is: " + str(acc))


			print(class_rep)

			weights = weights[0]

			resTop = sorted(range(len(weights)), key=lambda sub: weights[sub])[-25:]
			resBottom = sorted(range(len(weights)), key=lambda sub: weights[sub])[:25]
			if not do_ngrams:
				model_name_amp = model_name + "_" + str(acc) + "_.sav"
				pickle.dump(net, open(model_name_amp, 'wb'))
			fout.write(class_rep)
			fout.write("\n")
			fout.write("Male Top Words: \n")
			for index in resTop:
				fout.write(cumulative_word_vec[index] + ' ' + str(float(weights[index])) + '\n')
			fout.write("Female Top Words: \n")
			for index in resBottom:
				fout.write(cumulative_word_vec[index] + ' ' + str(float(weights[index])) + '\n')
			'''
			if multiple:
				if do_ngrams == True:
					multiple = False
					if not pos:
						count_vectors_train = np.load("./store/word" + "_" + str(l) + "gram_CorpusCountstrain.npy")
						count_vectors_test = np.load("./store/pos" + "_" + str(l) + "gram_CorpusCountstest.npy")
						word_model = "./store/" + type + "_" + str(l) + "gram_vocab.npy"
						print(os.path.exists(word_model))
						cumulative_word_vec = np.load(word_model)
						fout = open('vocabulary/output_words_50Articles_wordgram' + str(l) +'.txt', 'w')

						model_name = str(l) + "wordgram"
					else:
						count_vectors_train = np.load("./store/pos" + "_" + str(l) + "gram_CorpusCountstrain.npy")
						count_vectors_test = np.load("./store/pos" + "_" + str(l) + "gram_CorpusCountstest.npy")

						fout = open('vocabulary/output_words_50Articles_posgram' + str(l) + '.txt', 'w')

						model_name = str(l) + "wordgram"
				net = SVM()
				print("training")
				net.Train(count_vectors_train, train_labels+validation_labels, count_vectors_train,
						  train_labels+validation_labels)  # no validation occurs here, so last 2 params do nothing
				weights = net.Get_Weights()
				predictions = net.Predict(count_vectors_test)
				acc = accuracy_score(test_labels, predictions)
				class_rep = classification_report(test_labels, predictions)
				target_names = ['Female', 'Male']
				print("accuracy is: " + str(acc))

				# if the accuracy is high enough, print the metrics, and print top words to a file
				# if acc >= 0.60:

				print(class_rep)

				weights = weights[0]

				resTop = sorted(range(len(weights)), key=lambda sub: weights[sub])[-25:]
				resBottom = sorted(range(len(weights)), key=lambda sub: weights[sub])[:25]
				model_name_amp = model_name + "_" + str(acc) + ".sav"
				if not do_ngrams:
					pickle.dump(net, open(model_name_amp, 'wb'))
			if do_ngrams == False:
				if not_pos and balanced:
					fout = open('vocabulary/output_words_50Articles_allwords.txt', 'w')
				if not not_pos and balanced:
					fout = open('vocabulary/output_words_50Articles_adj.txt', 'w')
				if not_pos and not balanced:
					fout = open('vocabulary/output_words_allArticles_allwords.txt', 'w')
				if not not_pos and not balanced:
					fout = open('vocabulary/output_words_allArticles_adj.txt', 'w')

			fout.write(class_rep)
			fout.write("\n")
			fout.write("Male Top Words: \n")
			for index in resTop:
				fout.write(cumulative_word_vec[index] + ' ' + str(float(weights[index])) + '\n')
			fout.write("Female Top Words: \n")
			for index in resBottom:
				fout.write(cumulative_word_vec[index] + ' ' + str(float(weights[index])) + '\n')
			'''


from Models.SVM_engine import SVM
from Models.KNN_engine import KNN
from Models.Naive_Bayes_engine import Naive_Bayes
from Models.Linear_Classification_engine import Linear_Classifier 
from Models.NN_engine import NN
from Models.NN_engine import  Linear_NN

from DataReader import DataReader
import ApplicationConstants
from doc2vec import doc
from Metrics import Metrics

#This class is used for training various models on the embedding data. 
#what this class is NOT: visualization, sentiment training, or anything outside of utilizing the models
class ModelOrchestration():
	
	def __init__(self):
		
		self.docEmbed = doc()
		self.Metrics = Metrics()

	# trains and returns the vector embeddings for doc2vec or sent2vec
	#	Parameters:5
	#	articles: a list of articles that are cleaned
	#	labels: a list of labels corresponding to the article genders
	def embed_fold(self, articles, labels, fold, leaning):

		model = self.docEmbed.Embed(articles, labels)
		targets, regressors = self.docEmbed.gen_vec(model, articles, labels)

		return list(targets), regressors, model

	# trains all models against all leanings
	#	
	#	Parameters: 
	#	------------
	#	splits: A list of the splits 
	#
	def train_all(self, splits):

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
					prediction = model.Predict(test_embeddings)
					
					print("Model:", str(type(model)).split('.')[2].split('\'')[0], "precision:", self.Metrics.Precision(prediction, test_labels), "recall:", self.Metrics.Recall(prediction, test_labels), "F-Measure:", self.Metrics.Fmeasure(prediction, test_labels))   

if __name__ == "__main__":
	orchestration = ModelOrchestration()
	reader = DataReader()

	#dirty data first
	dirty_splits = reader.Load_Splits(ApplicationConstants.all_articles_random_v2, None, number_of_articles=50, clean=False, save=False, shouldRandomize=False)
	orchestration.train_all(dirty_splits)

	#clean data
	dirty_splits = reader.Load_Splits(ApplicationConstants.all_articles_random_v2_cleaned, None, number_of_articles=50, clean=False, save=False, shouldRandomize=False)
	orchestration.train_all(dirty_splits)
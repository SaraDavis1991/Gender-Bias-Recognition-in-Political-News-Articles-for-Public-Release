
from DataReader import DataReader
import ApplicationConstants

#This class is used to generate the most descriptive word list for a given set of empeddings. 
class Bow():

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

		if os.path.isfile('np_cum_vec.npy'):
			numpy_cumulative = np.load('np_cum_vec.npy')
			cumulative_word_vec = numpy_cumulative.tolist()
		else:
			cumulative_word_vec = orchestrator.calc_word_vector(articles)
			numpy_cumulative = np.array(cumulative_word_vec)
			np.save('np_cum_vec.npy', numpy_cumulative)

			if verbose:
				print("total num words = " + str(len(cumulative_word_vec)))

		return cumulative_word_vec

	def generate_bow(self, splits, verbose=True): 

		cumulative_word_vec = self.load_cumulative_word_vec(splits, verbose)

		articles_list = list(map(lambda article: article.Content, articles))
		labels = list(map(lambda article: article.Label.TargetGender,articles))

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

		if os.path.isfile('np_count_vec.npy'):
			numpy_cumulative = np.load('np_count_vec.npy')
			count_vectors = numpy_cumulative.tolist()
		else:
			for article in list_articles:
				count_vectors.append(orchestrator.calc_count_doc_count_vector(cumulative_word_vec, article))
		   
			numpy_count = np.array(count_vectors)
			np.save('np_count_vec.npy', numpy_count)

		trainLen = int(len(count_vectors)*0.8)
		
		if verbose: print("building net")

		net = SVM()

		if verbose: print("training")
		weights = net.Train(count_vectors[:trainLen], list_labels[:trainLen], count_vectors[trainLen:], list_labels[trainLen:])

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

		pickle.dump(net, open("perceptron2.sav", 'wb'))

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
	
	#dirty data first
	dirty_splits = reader.Load_Splits(ApplicationConstants.all_articles_random_v2, None, number_of_articles=50, clean=False, save=False, shouldRandomize=False)
	cumulative_word_vec, weights, resTop, resBottom = bow.generate_bow(dirty_splits)
	bow.print_bow(cumulative_word_vec, weights, resTop, resBottom)

	#clean data
	dirty_splits = reader.Load_Splits(ApplicationConstants.all_articles_random_v2_cleaned, None, number_of_articles=50, clean=False, save=False, shouldRandomize=False)

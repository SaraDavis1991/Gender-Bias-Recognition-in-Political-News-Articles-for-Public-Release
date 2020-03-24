from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np 

from doc2vec import doc
from DataReader import DataReader
import ApplicationConstants

cmap = ['red','blue']

class Visualizer():

	def __init__(self):
		
		self.docEmbed = doc()

	def hover(self, event):
		vis = self.annot.get_visible()
		if event.inaxes == self.ax:
			cont, ind = self.sc.contains(event)
			if cont:
				self.update_annot(ind)
				self.annot.set_visible(True)
				self.fig.canvas.draw_idle()
			else:
				if vis:
					self.annot.set_visible(False)
					self.fig.canvas.draw_idle()

	def update_annot(self, ind):

		pos = self.sc.get_offsets()[ind["ind"][0]]
		self.annot.xy = pos
		text = "{}".format(" ".join([self.articles[n].Title for n in ind["ind"]]))
		self.annot.set_text(text)
	   # self.annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
		self.annot.get_bbox_patch().set_alpha(0.4) 

	def plot_TSNE(self, leaning, weights, true_labels, articles):
		
		self.articles = articles

		self.genders = list(map(lambda label: 'Male' if label == 1 else 'Female', true_labels))
		tsne = TSNE(verbose=1)
		results = tsne.fit_transform(weights)

		self.fig, self.ax = plt.subplots()

		self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
					bbox=dict(boxstyle="round", fc="w"),
					arrowprops=dict(arrowstyle="->"))
		self.annot.set_visible(False)

		#self.sc = plt.scatter(x=results[:,0], y=results[0:,1], c=true_labels, cmap=matplotlib.colors.ListedColormap(cmap))
		self.sc = sns.scatterplot(x=results[:,0], y=results[0:,1], palette=sns.color_palette("hls", 2), hue=self.genders)

		#plt.setp(ax.get_legend().get_texts(), fontsize='40')
		plt.legend( loc='best', prop={'size': 15})
		#plt.legend(*self.sc.legend_elements(), loc='best', prop={'size': 20})
		plt.title('t-SNE Article Distribution for ' + leaning, fontsize=20)
		#self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
		plt.show()
		#plt.savefig("visualizations/" + leaning)

	def graph_sentiment(self, Fsentiment, Msentiment):

		pos_counts_per_leaning_female = [] 
		neg_counts_per_leaning_male = []
		pos_counts_per_leaning_male = [] 
		neg_counts_per_leaning_female = []
		leanings = ["Huffpost", "New York Times", "USA Today", "Fox", "Breitbart"]

		breitbart_female_sentiments = list(map(lambda sentiment: (sentiment[1], sentiment[2]), list(filter(lambda leaning: leaning[0] == ApplicationConstants.Breitbart, Fsentiment))))
		breitbart_male_sentiments = list(map(lambda sentiment: (sentiment[1], sentiment[2]), list(filter(lambda leaning: leaning[0] == ApplicationConstants.Breitbart, Msentiment))))
		fox_female_sentiments = list(map(lambda sentiment: (sentiment[1], sentiment[2]), list(filter(lambda leaning: leaning[0] == ApplicationConstants.Fox, Fsentiment))))
		fox_male_sentiments = list(map(lambda sentiment: (sentiment[1], sentiment[2]), list(filter(lambda leaning: leaning[0] == ApplicationConstants.Fox, Msentiment))))
		usa_female_sentiments = list(map(lambda sentiment: (sentiment[1], sentiment[2]), list(filter(lambda leaning: leaning[0] == ApplicationConstants.usa_today, Fsentiment))))
		usa_male_sentiments = list(map(lambda sentiment: (sentiment[1], sentiment[2]), list(filter(lambda leaning: leaning[0] == ApplicationConstants.usa_today, Msentiment))))
		nyt_female_sentiments = list(map(lambda sentiment: (sentiment[1], sentiment[2]), list(filter(lambda leaning: leaning[0] == ApplicationConstants.New_york_times, Fsentiment))))
		nyt_male_sentiments = list(map(lambda sentiment: (sentiment[1], sentiment[2]), list(filter(lambda leaning: leaning[0] == ApplicationConstants.New_york_times, Msentiment))))
		hp_female_sentiments = list(map(lambda sentiment: (sentiment[1], sentiment[2]), list(filter(lambda leaning: leaning[0] == ApplicationConstants.HuffPost, Fsentiment))))
		hp_male_sentiments = list(map(lambda sentiment: (sentiment[1], sentiment[2]), list(filter(lambda leaning: leaning[0] == ApplicationConstants.HuffPost, Msentiment))))

		male_leanings = [hp_male_sentiments, nyt_male_sentiments, usa_male_sentiments, fox_male_sentiments, breitbart_male_sentiments]
		female_leanings = [hp_female_sentiments, nyt_female_sentiments, usa_female_sentiments, fox_female_sentiments, breitbart_female_sentiments]
   
		for leaning in range(5): 

			femaleVals = []
			maleVals = []
			
			for sentiment in female_leanings[leaning]:
				result = self.calc_sent(sentiment[0], sentiment[1])

				if result is not None:
					femaleVals.append(self.calc_sent(sentiment[0], sentiment[1]))

			for sentiment in male_leanings[leaning]:
				result = self.calc_sent(sentiment[0], sentiment[1])

				if result is not None:
					maleVals.append(result)

			male_articles_length = len(male_leanings[leaning])
			female_articles_length = len(female_leanings[leaning])

			female_pos = len(list(filter(lambda sent: sent == 'pos', femaleVals)))
			female_neg = len(list(filter(lambda sent: sent == 'neg', femaleVals)))
			male_pos = len(list(filter(lambda sent: sent == 'pos', maleVals)))
			male_neg = len(list(filter(lambda sent: sent == 'neg', maleVals)))

			print(leanings[leaning])
			print("Num Female Pos: " + str(female_pos))
			print("Num Female Neg: " + str(female_neg))
			print("Num Male Pos: " + str(male_pos))
			print("Num Male Neg: " + str(male_neg))

			pos_counts_per_leaning_female.append(female_pos / female_articles_length)
			neg_counts_per_leaning_female.append(female_neg / female_articles_length)
			neg_counts_per_leaning_male.append(male_neg / male_articles_length)
			pos_counts_per_leaning_male.append(male_pos / male_articles_length)

		plt.plot(leanings, pos_counts_per_leaning_female, marker='D', label='Positive Female Articles', color='seagreen')
		plt.plot(leanings, neg_counts_per_leaning_female, marker='D', label='Negative Female Articles', color='slateblue')
		plt.plot(leanings, pos_counts_per_leaning_male, marker='D', label='Positive Male Articles', color='orange')
		plt.plot(leanings, neg_counts_per_leaning_male, marker='D', label='Negative Male Articles', color='crimson')

		plt.ylabel('Mean Leaning Sentiment Positive:Negative Ratio')
		plt.title('Positive and Negative Sentiment by Leaning and Gender')
		plt.xticks(leanings)
		plt.ylim((0, 1))
		plt.legend(loc='center right')
		plt.show()

	def calc_sent(self, sentiment, confidence):

		if (abs(confidence) < 0.25):
		   return None

		if sentiment == 0:
			return 'neg'
		else:
			return 'pos'
		#score = sentiment[0]
		#magnitude = sentiment[1]

		#if score > 0.25:
		#    return 'pos'
		#elif score < -0.25:
		#    return 'neg'

	def run_visualization(self, splits):
		''' trains all models against all leanings
		
		Parameters: 
		------------
		splits: A list of the splits 

		''' 

		#loop over all leanings
		for leaning in splits[0]:

			print("For leaning:", leaning.upper())

			#train embeddings
			training_dataset = splits[0][leaning][ApplicationConstants.Train]
			validation_dataset = splits[0][leaning][ApplicationConstants.Validation]
			test_dataset = splits[0][leaning][ApplicationConstants.Test]           
			article_labels, article_embeddings, article_model = self.docEmbed.embed_fold(list(map(lambda article: article.Content, training_dataset + validation_dataset + test_dataset)), list(map(lambda article: article.Label.TargetGender, training_dataset + validation_dataset + test_dataset)), leaning)

			#training embeddings
			training_embeddings = article_embeddings[:len(training_dataset)]
			training_labels = article_labels[:len(training_dataset)]

			#validation embeddings 
			validation_embeddings = article_embeddings[len(training_dataset): len(training_dataset) + len(validation_dataset)]
			validation_labels = article_labels[len(training_dataset): len(training_dataset) + len(validation_dataset)]

			#test embeddings
			test_embeddings = article_embeddings[len(training_dataset) + len(validation_dataset):]
			test_labels = article_labels[len(training_dataset) + len(validation_dataset):]

			#model = models[0] 
			#model.Model.coefs_[model.Model.n_layers_ - 2]
			self.plot_TSNE(leaning, training_embeddings + validation_embeddings + test_embeddings, training_labels + validation_labels + test_labels, training_dataset + validation_dataset + test_dataset)

if __name__ == "__main__":
	visualizer = Visualizer()
	reader = DataReader()

	dirty = reader.Load_Splits(ApplicationConstants.all_articles_random_v2, None, number_of_articles=500, clean=False, save=False, shouldRandomize=False)
	splits = reader.Load_Splits(ApplicationConstants.all_articles_random_v2_cleaned, None, number_of_articles=50, clean=False, save=False, shouldRandomize=False)
	visualizer.run_visualization(dirty)
	visualizer.run_visualization(splits)
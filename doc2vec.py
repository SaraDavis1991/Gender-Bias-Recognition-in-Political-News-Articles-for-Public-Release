
import ApplicationConstants
from DataReader import DataReader
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing

#print(Orchestrator.sources['breitbart'].Articles[58].Label.TargetName, Orchestrator.sources['breitbart'].Articles[88].Label.TargetName, Orchestrator.sources['breitbart'].Articles[37].Label.TargetName, Orchestrator.sources['breitbart'].Articles[29].Label.TargetName)
'''
articles = []
dr = DataReader()
sources = dr.Load(ApplicationConstants.all_articles)

for i in range(len(sources['breitbart'].Articles)):
	if sources['breitbart'].Articles[i].Label.TargetName != ApplicationConstants.ElizabethWarren:
		c = sources['breitbart'].Articles[i].Content
		articles.append(c)
articles = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(articles)]
'''
class doc():
	def __init__(self):
		print("butt")
	
	def Embed(self, cleaned_with_labels):
		tagged = data.apply(lambda r: TaggedDocument(words = cleaned_with_labels[0], tags = cleaned_with_labels[1]), axis= -1)
		model = Doc2Vec(size = 300, alpha = 0.001, min_alpha = 0.00025, min_count = 2, dm = 0, negative = 5, hs = 0, sample = 0, workers = multiprocessing.cpu_count()) #dm 1 is pv-dm, dm 0 is pv-dbow size is feature vec size, alpha is lr, negative is noise words, sample is thresh for down smample
		model.build_vocab(articles)

		for epoch in range(100):
			print('iteration {0}'.format(epoch))
			model.train(articles, total_examples = model.corpus_count , epochs= model.iter)
			model.alpha -= 0.0002
			model.min_alpha = model.alpha
		sents = tagged.values
		targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps = 20)) for doc in sents])
		return targets, regressors
		#model.save("d2v.model")
		#print("model saved")
		


'''
articles2 = []
for i in range(len(sources['breitbart'].Articles)):
	if sources['breitbart'].Articles[i].Label.TargetName == ApplicationConstants.ElizabethWarren:
		c = sources['breitbart'].Articles[i].Content
		articles2.append(c)
#articles2= word_tokenize(_d.lower() for _d in enumerate(articles)) 
#print(articles2)
v1 = model.infer_vector(articles2)
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)
print(model.docvecs['1'])
'''


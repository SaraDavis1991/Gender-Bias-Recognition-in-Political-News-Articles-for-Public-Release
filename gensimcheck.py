import Orchestrator 
import ApplicationConstants
from DataReader import DataReader
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

#print(Orchestrator.sources['breitbart'].Articles[58].Label.TargetName, Orchestrator.sources['breitbart'].Articles[88].Label.TargetName, Orchestrator.sources['breitbart'].Articles[37].Label.TargetName, Orchestrator.sources['breitbart'].Articles[29].Label.TargetName)

articles = []
dr = DataReader()
sources = dr.Load(ApplicationConstants.all_articles)

for i in range(len(sources['breitbart'].Articles)):
	if sources['breitbart'].Articles[i].Label.TargetName != ApplicationConstants.ElizabethWarren:
		c = sources['breitbart'].Articles[i].Content
		articles.append(c)
articles = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(articles)]


model = Doc2Vec(size = 100, alpha = 0.001, min_alpha = 0.00025, min_count = 1, dm = 1)
model.build_vocab(articles)

for epoch in range(100):
	print('iteration {0}'.format(epoch))
	model.train(articles, total_examples = model.corpus_count , epochs= model.iter)
	model.alpha -= 0.0002
	model.min_alpha = model.alpha
model.save("d2v.model")
print("model saved")


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


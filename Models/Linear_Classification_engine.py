from Interfaces.IModel import IModel
from interface import implements

from sklearn.linear_model import SGDClassifier

class Linear_Classifier(implements(IModel)):
	
	def __init__(self):
		self.Model = self.__BuildModel()

	def __BuildModel(self):
		model = SGDClassifier(n_jobs=-1, loss = 'log')
		return model

	def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
		self.Model.fit(trainFeatures, trainLabels)
		
	def Predict(self, features): 
		return self.Model.predict(features)
	
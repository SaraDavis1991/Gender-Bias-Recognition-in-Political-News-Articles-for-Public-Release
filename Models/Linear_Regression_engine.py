from Interfaces.IModel import IModel
from interface import implements

from sklearn.linear_model import LinearRegression

class Linear_Regression(implements(IModel)):
	
	def __init__(self):
		self.Model = self.__BuildModel()

	def __BuildModel(self):
		model = LinearRegression(n_jobs=-1)
		return model

	def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):	
		self.Model.fit(trainFeatures, trainLabels)
		
	def Predict(self, features): 
		return self.Model.predict(features)
	
from Interfaces.IModel import IModel
from interface import implements

class Linear_Regression(implements(IModel)):
	
	def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
		pass 

	def Predict(self, features): 
		pass
	
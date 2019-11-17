from Interfaces.IModel import IModel
from interface import implements
from sklearn import metrics as met 
from sklearn.naive_bayes import GaussianNB
import numpy as np 
import pickle
import sys

from Metrics import Metrics

class Naive_Bayes(implements(IModel)):
	
	def __init__(self):
		self.Metrics = Metrics()
		self.model = GaussianNB()

	def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
		
		smooths = [0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]

		best_smooth = 0
		bestF1 = -1 

		for smooth in smooths:
			
			self.model.var_smoothing = smooth 
			self.model.fit(trainFeatures, trainLabels)
			
			preds = self.Predict(validationFeatures)
			currentF = self.Metrics.Fmeasure(preds, validationLabels)

			if currentF > bestF1:
				bestF1 = currentF
				best_smooth = smooth
		
		#reset the model to use the best smoothing vale
		self.model.var_smoothing = smooth
		self.model.fit(trainFeatures, trainLabels) 

	def Predict(self, features):
		return self.model.predict(features) 
from Interfaces.IModel import IModel
from interface import implements
from sklearn import metrics as met 
from sklearn.naive_bayes import GaussianNB
import numpy as np 
import pickle
import sys

class Naive_Bayes(implements(IModel)):
    
    def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
    	
    	smooths = [0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    	bestF1 = -1
    	for smooth in smooths:
    		self.model = GaussianNB(var_smoothing=smooth)
    		self.model.fit(trainFeatures, trainLabels)
    		preds = self.Predict(validationFeatures)
    		#print(preds)
    		#print(validationLabels)
    		tp, tn, fp, fn = self.Condition(preds, validationLabels)
    		#print("vals ", tp, tn, fp, fn)
    		recall = self.Recall(tp, fn)
    		precision = self.Precision(tp, fp)
    		currentF = self.Fmeasure(recall, precision)
    		#print(precision, recall)
    		self.Accuracy(tp, fp, fn, tn)
    		

    		if currentF > bestF1:
    			#print("Best F1: "+ str(currentF)+ " smoothing: " + str(smooth))
    			bestF1 = currentF  

    def Predict(self, features):
    	return self.model.predict(features), self.model.predict_proba(features)
        

    def Accuracy(self, tp, fp, fn, tn): 
    	if tp + fp > 0 and tp + fp + fn + tn > 0:
    		return (tp + tn) / (tp + fp+fn + tn)
    	else:
    		return 0
        

    def Fmeasure(self, recall, precision):
    	#print("P& R: ", precision, recall)

    	if precision > 0 and recall > 0:
    		mult = 2 * precision * recall
    		addit = precision + recall
    		#print(mult/addit)
    		return mult/addit
    	else:
    		return 0

    def Condition(self, prediction, truth_labels):
    	tp = 0
    	tn = 0
    	fp = 0
    	fn = 0
    	for i in range(len(prediction)):
    		#print(prediction[i], truth_labels[i])
    		if prediction[i]=="Female" and truth_labels[i] == "Female": 
    			tp+=1
    		if prediction[i] == "Male" and truth_labels[i] =="Male":
    			tn +=1
    		if prediction[i] == "Female" and truth_labels[i] != "Female":
    			fp +=1
    		if prediction[i] == "Male" and truth_labels[i] != "Male":
    			fn +=1
    	return tp, tn, fp, fn

    def Recall(self, tp, fn):
    	if tp > 0:
    		return (tp/(tp+fn))
    	else:
    		return 0

    def Precision(self, tp, fp):
    	if tp > 0:
    		return (tp/(tp+fp))
    	else:
    		return 0
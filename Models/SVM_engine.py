from Interfaces.IModel import IModel
from interface import implements

from sklearn.metrics import accuracy_score
from sklearn import svm

from Metrics import Metrics

class SVM(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_SVM()
        self.Metrics = Metrics()
    def Build_SVM(self):

        model = svm.SVC(gamma='auto', probability = False)
        return model 

    def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
        self.Model.fit(trainFeatures, trainLabels)

    def Predict(self, features): 
        
        prediction = self.Model.predict(features) 
        confidence = self.Model.decision_function(features)

        return prediction, confidence

from Interfaces.IModel import IModel
from interface import implements

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class NN(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_Model()

    def Build_Model(self):

        model = MLPClassifier()
        return model 

    def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
        self.Model.fit(trainFeatures, trainLabels)

    def Predict(self, features): 
        return self.Model.predict(features) 

    def Accuracy(self, prediction, truth_labels): 
        return accuracy_score(prediction, truth_labels) 
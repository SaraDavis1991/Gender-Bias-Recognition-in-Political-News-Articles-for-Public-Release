from Interfaces.IModel import IModel
from interface import implements

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class KNN(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_Model()

    def Build_Model(self):

        model = KNeighborsClassifier(n_neighbors=2)
        return model 

    def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
        self.Model.fit(trainFeatures, trainLabels)

    def Predict(self, features): 
        return self.Model.predict(features) 

    def Accuracy(self, prediction, truth_labels): 
        return accuracy_score(prediction, truth_labels) 
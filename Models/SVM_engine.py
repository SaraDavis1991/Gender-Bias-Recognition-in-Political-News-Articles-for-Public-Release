from Interfaces.IModel import IModel
from interface import implements

from sklearn.metrics import accuracy_score
from sklearn import svm

class SVM(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_SVM()

    def Build_SVM(self):

        model = svm.SVC(gamma='auto')

        #anything else? 

        return model 

    def Train(self, features, labels):
        
        self.Model.fit(features, labels)

    def Predict(self, features): 
        
        prediction = self.Model.predict(features) 
        return prediction

    def Accuracy(self, prediction, truth_labels): 
        return accuracy_score(truth_labels, prediction)
    
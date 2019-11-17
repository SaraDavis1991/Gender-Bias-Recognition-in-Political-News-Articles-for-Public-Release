from Interfaces.IModel import IModel
from interface import implements

class Linear_Regression(implements(IModel)):
    
    def Train(self, features, labels):
        pass 

    def Predict(self, features): 
        pass

    def Accuracy(self, prediction, truth_labels): 
        pass 
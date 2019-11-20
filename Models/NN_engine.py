from Interfaces.IModel import IModel
from interface import implements

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.neural_network import MLPClassifier


class NN(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_Model()

    def Build_Model(self):

        model = MLPClassifier(hidden_layer_sizes=(195,))
        return model 

    def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
        self.Model.fit(trainFeatures, trainLabels)

    def Predict(self, features): 
        return self.Model.predict(features) 
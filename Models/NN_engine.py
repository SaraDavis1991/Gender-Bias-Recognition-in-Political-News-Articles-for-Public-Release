from Interfaces.IModel import IModel
from interface import implements

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron


class NN(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_Model()

    def Build_Model(self):

        model = MLPClassifier()
        return model 

    def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
        self.Model.fit(trainFeatures, trainLabels)
        weights = self.Model.coefs_

        return weights

    def Predict(self, features): 
        return self.Model.predict(features)

class Linear_NN(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_Model()

    def Build_Model(self):
        #eta0 = 1
        model = Perceptron(alpha = 0.0001, n_jobs = -1, early_stopping = True, validation_fraction = .2, n_iter_no_change = 100, max_iter = 10000, verbose = 10, penalty = 'l1')

        #return weights

        return model

    def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
        print("fitting")
        self.Model.fit(trainFeatures, trainLabels)
        print("weights")
        weights = self.Model.coef_


        return weights

    def Predict(self, features):
        return self.Model.predict(features)
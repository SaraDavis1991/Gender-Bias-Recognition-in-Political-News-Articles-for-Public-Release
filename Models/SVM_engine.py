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

        model = svm.SVC(gamma='auto')
        return model 

    def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
        
        gammas = ['auto', 'scale']
        best_gamma = ''
        best_f = -1 

        for gamma in gammas: 

            self.Model.gamma = gamma
            self.Model.fit(trainFeatures, trainLabels)

            prediction = self.Model.predict(validationFeatures)
            f_measure = self.Metrics.Fmeasure(prediction, validationLabels)

            if (f_measure > best_f):
                best_f = f_measure
                best_gamma = gamma

        #reset model with best gamma
        self.Model.gamma = best_gamma
        self.Model.fit(trainFeatures, trainLabels)

    def Predict(self, features): 
        
        prediction = self.Model.predict(features) 
        return prediction
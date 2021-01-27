from Interfaces.IModel import IModel
from interface import implements

from sklearn.metrics import accuracy_score
from sklearn import svm
import sklearn.preprocessing as preprocessing

from Metrics import Metrics

class SVM(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_SVM()
        self.Metrics = Metrics()
        #self.min_max_scaler = preprocessing.MinMaxScaler()
    def Build_SVM(self):
        #57 at 300
        model = svm.SVC(kernel='linear', gamma='auto', C = 1, probability = False, tol = 0.5, max_iter = 300,  shrinking = True )
        return model 

    def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):
        #trainFeatures = self.min_max_scaler.fit_transform(trainFeatures)
        #print(trainFeatures)
        self.Model.fit(trainFeatures, trainLabels)

    def Predict(self, features, shouldPredictConfidences=False): 
        #features = self.min_max_scaler.transform(features)
        #print(features)
        prediction = self.Model.predict(features) 

        if (shouldPredictConfidences):
            confidence = self.Model.decision_function(features) 

            return prediction, confidence

        return prediction

    def Get_Weights(self):
        return self.Model.coef_

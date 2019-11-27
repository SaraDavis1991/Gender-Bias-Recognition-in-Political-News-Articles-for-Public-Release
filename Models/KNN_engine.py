from Interfaces.IModel import IModel
from interface import implements

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from Metrics import Metrics

class KNN(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_Model()
        self.Metrics = Metrics() 

    def Build_Model(self):

        model = KNeighborsClassifier()
        return model 

    def Train(self, trainFeatures, trainLabels, validationFeatures, validationLabels):

        n_neighbors = [4, 2, 4, 6, 8]
        best_f = -1 
        best_neighbor = -1

        for neighbor in n_neighbors:
            self.Model.n_neighbors = neighbor
            self.Model.fit(trainFeatures, trainLabels)

            prediction = self.Model.predict(validationFeatures)
            f_measure = self.Metrics.Fmeasure(prediction, validationLabels)

            if (f_measure > best_f):
                best_f = f_measure
                best_neighbor = neighbor

        #reset model with best neighbor

        print("best nn for KNN", best_neighbor)
        self.Model.n_neighbors = best_neighbor
        self.Model.fit(trainFeatures, trainLabels)

    def Predict(self, features): 
        return self.Model.predict(features) 
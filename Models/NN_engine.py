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

    def Accuracy(self, tp, fp, fn, tn): 
        if tp + fp > 0 and tp + fp + fn + tn > 0:
            return (tp + tn) / (tp + fp+fn + tn)
        else:
            return 0
        

    def Fmeasure(self, recall, precision):
        #print("P& R: ", precision, recall)

        if precision > 0 and recall > 0:
            mult = 2 * precision * recall
            addit = precision + recall
            #print(mult/addit)
            return mult/addit
        else:
            return 0

    def Condition(self, prediction, truth_labels):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(prediction)):
            #print(prediction[i], truth_labels[i])
            if prediction[i]=="Female" and truth_labels[i] == "Female": 
                tp+=1
            if prediction[i] == "Male" and truth_labels[i] =="Male":
                tn +=1
            if prediction[i] == "Female" and truth_labels[i] != "Female":
                fp +=1
            if prediction[i] == "Male" and truth_labels[i] != "Male":
                fn +=1
        return tp, tn, fp, fn

    def Recall(self, tp, fn):
        if tp > 0:
            return (tp/(tp+fn))
        else:
            return 0

    def Precision(self, tp, fp):
        if tp > 0:
            return (tp/(tp+fp))
        else:
            return 0
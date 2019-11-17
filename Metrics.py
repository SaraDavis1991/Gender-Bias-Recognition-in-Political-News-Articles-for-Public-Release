
from sklearn.metrics import accuracy_score

class Metrics():

    def Accuracy(self, prediction, labels): 
       return accuracy_score(labels, prediction)
        
    def Fmeasure(self, prediction, labels):

        precision = self.Precision(prediction, labels)
        recall = self.Recall(prediction, labels) 

        if precision > 0 and recall > 0:

            mult = 2 * precision * recall
            addit = precision + recall

            return mult/addit
        else:
            return 0

    def Recall(self, prediction, labels):
        tp, _, _, fn = self.__Condition(prediction, labels)
        
        if tp > 0:
            return (tp/(tp+fn))
        else:
            return 0

    def Precision(self, prediction, labels):
        tp, _, fp, _ = self.__Condition(prediction, labels)

        if tp > 0:
            return (tp/(tp+fp))
        else:
            return 0

    def __Condition(self, prediction, truth_labels):

        tp, tn, fp, fn = 0, 0, 0, 0

        for i in range(len(prediction)):

            if prediction[i] == "Female" and truth_labels[i] == "Female": 
                tp+=1
            if prediction[i] == "Male" and truth_labels[i] =="Male":
                tn +=1
            if prediction[i] == "Female" and truth_labels[i] != "Female":
                fp +=1
            if prediction[i] == "Male" and truth_labels[i] != "Male":
                fn +=1
        return tp, tn, fp, fn
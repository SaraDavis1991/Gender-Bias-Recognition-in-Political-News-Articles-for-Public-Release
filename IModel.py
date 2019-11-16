from interface import Interface

class IModel(Interface):
    '''The interface definition for models ''' 

    def Train(self, features, labels):
        ''' Trains a given model on features and labels '''
        pass 

    def Predict(self, features, labels):
        ''' Predicts on test data for a given model '''
        pass

    
from DataReader import DataReader

from preprocessor import Preprocessor
from DataContracts import Article
import ApplicationConstants

class Orchestrator():

    def __init__(self):
        self.Reader = DataReader()
        self.Preprocessor = Preprocessor()
        self.Splits = None 
        
    def read_data(self):       
        self.Splits = self.Reader.Load_Splits(ApplicationConstants.all_articles)

    def clean_all(self, data):
        contents = []
        for article in data:
            content = article.Content
            cleaned_content = orchestrator.Preprocessor.Clean(content)
            contents.append(cleaned_content)

        return contents


orchestrator = Orchestrator()
data = orchestrator.read_data() 
contents = orchestrator.clean_all(data)


print (contents)


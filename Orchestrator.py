from DataReader import DataReader
from preprocessor import Preprocessor
from DataContracts import Article
import ApplicationConstants

class Orchestrator():

    def __init__(self):
        self.Reader = DataReader()
        self.Preprocessor = Preprocessor()
        self.Sources = None 
        
    def read_data(self):       
        self.Sources = self.Reader.Load(ApplicationConstants.all_articles)

        #need to pull nyt when fixed
        breitbart = sources[ApplicationConstants.Breitbart].Articles
        fox = sources[ApplicationConstants.Fox].Articles 
        usa = sources[ApplicationConstants.usa_today].Articles
        huffpost = sources[ApplicationConstants.HuffPost].Articles

        return fox + usa + breitbart + huffpost # + nyt

    def clean_all(self, data):
        contents = []
        for article in data:
            content = article.Content
            cleaned_content = orchestrator.Preprocessor.Clean(content)
            contents.append(cleaned_content)


orchestrator = Orchestrator()
data = orchestrator.read_data() 
cleaned = orchestrator.clean_all(data)


print (contents)
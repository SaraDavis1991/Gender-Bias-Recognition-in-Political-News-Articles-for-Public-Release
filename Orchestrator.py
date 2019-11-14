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
        breitbart = self.Sources[ApplicationConstants.Breitbart].Articles
        fox = self.Sources[ApplicationConstants.Fox].Articles 
        usa = self.Sources[ApplicationConstants.usa_today].Articles
        huffpost = self.Sources[ApplicationConstants.HuffPost].Articles

        return fox + usa + breitbart + huffpost # + nyt

    def clean_all(self, data):
        contents = []
        for article in data:
            content = article.Content
            cleaned_content = orchestrator.Preprocessor.Clean(content)
            contents.append(cleaned_content)

            print ("\n\n")
            print (content)

            print ("\n\n\n\n")

            print(cleaned_content)

        return contents


orchestrator = Orchestrator()
data = orchestrator.read_data() 
contents = orchestrator.clean_all(data)


print (contents)
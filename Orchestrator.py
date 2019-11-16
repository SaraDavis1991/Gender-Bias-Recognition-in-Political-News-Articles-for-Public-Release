from DataReader import DataReader

from preprocessor import Preprocessor
from DataContracts import Article
from doc2vec import doc
import ApplicationConstants

class Orchestrator():

    def __init__(self):
        self.Reader = DataReader()
        self.Preprocessor = Preprocessor()
        self.Splits = None 
        self.Sources = None 
        self.docEmbed = doc()
        
    def read_data(self):       
        return self.Reader.Load_Splits(ApplicationConstants.all_articles)

    def clean_all(self, data):
        contents = []
        for article in data:
            content = article.Content
            cleaned_content = orchestrator.Preprocessor.Clean(content)
            contents.append(cleaned_content)

        return contents
    
    def embed_fold(self, data):
        
        targets, regressors = self.docEmbed.Embed(data)
        print(targets)

        return targets



orchestrator = Orchestrator()
data = orchestrator.read_data() 
contents = orchestrator.clean_all(data)
embed = orchestrator.embed_fold(contents)


print (contents)


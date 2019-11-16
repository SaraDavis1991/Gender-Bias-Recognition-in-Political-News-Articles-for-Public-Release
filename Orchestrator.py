from DataReader import DataReader

from preprocessor import Preprocessor
from DataContracts import Article
from doc2vec import doc
import ApplicationConstants

class Orchestrator():

    def __init__(self):
        self.Reader = DataReader()
        self.Preprocessor = Preprocessor()
        self.Sources = None 
        self.docEmbed = doc()
        
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

        return contents
    
    def embed_fold(self, data):
        
        targets, regressors = orchestrator.docEmbed.Embed(data)
        print(targets)



orchestrator = Orchestrator()
data = orchestrator.read_data() 
contents = orchestrator.clean_all(data)
embed = orchestrator.embed_fold(contents)


print (contents)


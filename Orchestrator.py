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

    def clean_all(self, splits):

        for key in splits:

            data = splits[key]
            contents = []
            
            for article in data:

                content = article.Content
                cleaned_content = orchestrator.Preprocessor.Clean(content)
                contents.append((cleaned_content, article.Label.TargetGender))

        return contents
    
    def embed_fold(self, cleaned_with_labels):
        
        targets, regressors = self.docEmbed.Embed(cleaned_with_labels)
        print(targets)

        return targets



orchestrator = Orchestrator()
splits = orchestrator.read_data() 
cleaned_with_labels = orchestrator.clean_all(splits)
embed = orchestrator.embed_fold(cleaned_with_labels)


print (contents)


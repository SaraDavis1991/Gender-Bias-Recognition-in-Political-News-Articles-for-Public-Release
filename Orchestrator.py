from DataReader import DataReader
from preprocessor import Preprocessor
from DataContracts import Article
import ApplicationConstants

class Orchestrator():

    def __init__(self):
        self.Reader = DataReader()
        self.Preprocessor = Preprocessor()

    def read_data(self):       
        sources = self.Reader.Load(ApplicationConstants.all_articles)

        #need to pull nyt when fixed
        breitbart = sources[ApplicationConstants.Breitbart].Articles
        fox = sources[ApplicationConstants.Fox].Articles 
        usa = sources[ApplicationConstants.usa_today].Articles
        huffpost = sources[ApplicationConstants.HuffPost].Articles

        return fox + usa + breitbart + huffpost # + nyt

orchestrator = Orchestrator()
data = orchestrator.read_data() 

contents = []
for article in data:
    content = article.Content
    cleaned_content = orchestrator.Preprocessor.Remove_pos(content)
    contents.append(cleaned_content)

    print(content)
    print('\n\n///////////////////////////////////////\n\n')
    print(cleaned_content)

print (contents)
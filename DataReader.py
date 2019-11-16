import json
from DataContracts import Article
from DataContracts import Label
from DataContracts import Source
from collections import namedtuple
from typing import List
import ApplicationConstants

class DataReader():
    ''' This class is used to read and create json driven objects. ''' 

    def object_decoder(sel
    qf, obj): 
        if 'author' in obj:
            return Article(obj['title'], obj['url'], obj['subtitle'], obj['author'], obj['content'], obj['date'], obj['labels'])
        elif 'author_gender' in obj:
            return Label(obj['author_gender'], obj['target_gender'], obj['target_affiliation'], obj['target_name'])
        elif 'articles' in obj:
            return Source(obj['articles'])
        return obj
    
    def Load(self, filePath) -> List[Source]:
        with open(filePath, 'r') as read_file:
            data = json.load(read_file, object_hook=self.object_decoder)
        return data

    def Load_Splits(self, filePath):

        splits = {}

        #read the freaking json
        with open(filePath, 'r') as read_file:
            data = json.load(read_file, object_hook=self.object_decoder)

        #separate per source
        breitbart = data[ApplicationConstants.Breitbart].Articles
        fox = data[ApplicationConstants.Fox].Articles 
        usa = data[ApplicationConstants.usa_today].Articles
        huffpost = data[ApplicationConstants.HuffPost].Articles

        sources = [breitbart, fox, usa, huffpost]

        for source in sources: 
            
            #gather all male sources
            male_articles = list(filter(lambda article: article.Label.TargetGender == ApplicationConstants.Male, source))

            #gather all female sources 
            female_articles = list(filter(lambda article: article.Label.TargetGender == ApplicationConstants.Female, source))

            #get 4 random of each category 
            

        return ["a"]
import json
from DataContracts import Article
from DataContracts import Label
from collections import namedtuple
from typing import List

class DataReader():
    ''' This class is used to read and create json driven objects. ''' 
    def object_decoder(self, obj): 
        if 'author' in obj:
            return Article(obj['title'], obj['url'], obj['subtitle'], obj['author'], obj['content'], obj['date'], obj['labels'])
        elif 'author_gender' in obj:
            return Label(obj['author_gender'], obj['target_gender'], obj['target_affiliation'], obj['target_name'])
        return obj
    
    def Load(self, filePath) -> List[Article]:
        with open(filePath, 'r') as read_file:
            data = json.load(read_file, object_hook=self.object_decoder)
        return data['articles']
    
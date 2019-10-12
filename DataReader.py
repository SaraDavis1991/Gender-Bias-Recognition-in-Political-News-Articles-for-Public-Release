import json
from collections import namedtuple

class DataReader():
    ''' This class is used to read and create json driven objects.
    ''' 
    
    def Load(self, filePath):
        with open(filePath, 'r') as read_file:
            data = json.load(read_file, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        return data 
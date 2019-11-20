import json
from DataContracts import Article
from DataContracts import Label
from DataContracts import Source
from collections import namedtuple
from typing import List
import ApplicationConstants
import copy

class DataReader():
    ''' This class is used to read and create json driven objects. ''' 

    def object_decoder(self, obj): 
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

        candidate_split_file_names = [ApplicationConstants.fold_1, ApplicationConstants.fold_2, ApplicationConstants.fold_3, ApplicationConstants.fold_4, ApplicationConstants.fold_5]

        split_list = [] 

        #read the freaking json
        with open(filePath, 'r') as read_file:
            data = json.load(read_file, object_hook=self.object_decoder)

        #separate per source
        breitbart = data[ApplicationConstants.Breitbart].Articles
        fox = data[ApplicationConstants.Fox].Articles 
        usa = data[ApplicationConstants.usa_today].Articles
        huffpost = data[ApplicationConstants.HuffPost].Articles

        sources = [(ApplicationConstants.Breitbart, breitbart), (ApplicationConstants.Fox, fox), (ApplicationConstants.usa_today, usa), (ApplicationConstants.HuffPost, huffpost)]
   
        #loop over each split 
        for split_file_name in candidate_split_file_names: 

            training_candidates = []
            validation_candidates = []
            test_candidates = []
            split = {}

            #open split file 
            with open(split_file_name, 'r') as split_read:
                split_info = split_read.read()

            #parse the split 
            groups = split_info.split('\n')

            for group in groups:
                candidate_group_mapping = group.split(' ')

                #partition groups
                if (len(candidate_group_mapping) == 3):

                    #need to lower these to match the json data
                    candidate_group_mapping[0] = candidate_group_mapping[0].lower()
                    candidate_group_mapping[1] = candidate_group_mapping[1].lower()

                    #training
                    if candidate_group_mapping[2] == '0': 
                        training_candidates.append(candidate_group_mapping[0] + "_" + candidate_group_mapping[1])
                    #validation
                    elif candidate_group_mapping[2] == '1': 
                        validation_candidates.append(candidate_group_mapping[0] + "_" + candidate_group_mapping[1])
                    #test
                    elif candidate_group_mapping[2] == '2': 
                        test_candidates.append(candidate_group_mapping[0] + "_" + candidate_group_mapping[1])

            #loop over all sources
            for source_tuple in sources: 
                
                source_name = source_tuple[0]
                source = copy.deepcopy(source_tuple[1])
                split[source_name] = {}
                
                #get the training data defined by the split 
                split[source_name][ApplicationConstants.Train] = list(filter(lambda article: article.Label.TargetName.lower() in training_candidates, source))

                #get the validation data defined by the split
                split[source_name][ApplicationConstants.Validation] = list(filter(lambda article: article.Label.TargetName.lower() in validation_candidates, source))

                #get the test data define by the split  
                split[source_name][ApplicationConstants.Test] =list(filter(lambda article: article.Label.TargetName.lower() in test_candidates, source))

      
            split_list.append(split) 
            
        return split_list

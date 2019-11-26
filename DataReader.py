import json
import sys
from DataContracts import Article
from DataContracts import Label
from DataContracts import Source
from collections import namedtuple
from typing import List
import ApplicationConstants
import copy
from preprocessor import Preprocessor
import random

class DataReader():
    ''' This class is used to read and create json driven objects. ''' 

    def __init__(self):
        self.Preprocessor = Preprocessor()

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

    def Load_Splits(self, filePath, number_of_articles=50):

        candidate_split_file_names = [ApplicationConstants.fold_1, ApplicationConstants.fold_2, ApplicationConstants.fold_3, ApplicationConstants.fold_4, ApplicationConstants.fold_5]

        split_list = [] 

        #read the freaking json
        with open(filePath, 'r') as read_file:
            data = json.load(read_file, object_hook=self.object_decoder)

        #separate per source
        candidates = [ApplicationConstants.DonaldTrump, ApplicationConstants.JoeBiden, ApplicationConstants.JohnMccain, ApplicationConstants.BernieSanders, ApplicationConstants.BarrackObama, 
                      ApplicationConstants.HillaryClinton, ApplicationConstants.AlexandriaOcasioCortez, ApplicationConstants.BetsyDevos, ApplicationConstants.ElizabethWarren, ApplicationConstants.SarahPalin]

        breitbart = []
        fox = [] 
        usa = []
        huffpost = []
        nyt = [] 

        for candidate in candidates: 
            
            breitbart += list(filter(lambda article: article.Label.TargetName == candidate, data[ApplicationConstants.Breitbart].Articles))[:number_of_articles]
            fox += list(filter(lambda article: article.Label.TargetName == candidate, data[ApplicationConstants.Fox].Articles))[:number_of_articles]
            usa += list(filter(lambda article: article.Label.TargetName == candidate, data[ApplicationConstants.usa_today].Articles))[:number_of_articles]
            huffpost += list(filter(lambda article: article.Label.TargetName == candidate, data[ApplicationConstants.HuffPost].Articles))[:number_of_articles]
            nyt += list(filter(lambda article: article.Label.TargetName == candidate, data[ApplicationConstants.New_york_times].Articles))[:number_of_articles]

        sources = [(ApplicationConstants.Breitbart, breitbart), (ApplicationConstants.Fox, fox), (ApplicationConstants.usa_today, usa), (ApplicationConstants.HuffPost, huffpost), (ApplicationConstants.New_york_times, nyt)]
   
        for source_tuple in sources: 

            source_name = source_tuple[0]
            source = source_tuple[1]

            #candidates
            dt_breitbart = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.DonaldTrump, source))
            jb_breitbart = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.JoeBiden, source))
            bs_breitbart = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.BernieSanders, source))
            jm_breitbart = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.JohnMccain, source))
            bo_breitbart = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.BarrackObama, source))
            hc_breitbart = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.HillaryClinton, source))
            sp_breitbart = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.SarahPalin, source))
            aoc_breitbart = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.AlexandriaOcasioCortez, source))
            bd_breitbart = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.BetsyDevos, source))
            ew_breitbart = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.ElizabethWarren, source))

            print(source_name)
            print("trump:", len(dt_breitbart))
            print("joe biden:", len(jb_breitbart))
            print("bernie:", len(bs_breitbart))
            print("john:", len(jm_breitbart))
            print("obama:", len(bo_breitbart))
            print("hillary:", len(hc_breitbart))
            print("sarah:", len(sp_breitbart))
            print("aoc:", len(aoc_breitbart))
            print("betsy:", len(bd_breitbart))
            print("warren:", len(ew_breitbart))
            print("Cleaning data ", end='')
            sys.stdout.flush()

        #clean data 
        for source_index, source in enumerate(sources): 

            print(' . ', end='')
            sys.stdout.flush()

            #get article content
            articles = source[1]
            
            #clean, putting cleaned data back into the split dictionary
            for article_index, article in enumerate(articles):

                #convert labels to ints
                if (article.Label.TargetGender == ApplicationConstants.Female):
                    article.Label.TargetGender = 0
                elif (article.Label.TargetGender == ApplicationConstants.Male):
                    article.Label.TargetGender = 1

                content = article.Content
                cleaned_content = self.Preprocessor.Clean(content)
                sources[source_index][1][article_index].Content = cleaned_content 

        print("\nDone! \nStarting splitnig . . . ")

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
            
        print("Return splits . . . ")
        return split_list

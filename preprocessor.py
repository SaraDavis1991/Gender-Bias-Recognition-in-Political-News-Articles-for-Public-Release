
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import unicodedata
import string 
import re
import gensim.downloader as api
import StopWords
import json 

class Preprocessor():

    def Clean(self, data : str):
        ''' Removes POS that are NNP, PRP, or PRP$, and removes all stop words  '''

        #normalize the data, removing punctuation 
        data =  unicodedata.normalize('NFKC', data)
        
        #remove numbers
        data = re.sub('\d+', '', data)
        
        #remove "follow n on twitter"
        data = re.sub('(Follow)\s[a-zA-Z\s]*Twitter[a-zA-Z\s@]*.', '', data)

        #remove whole words from stop list 
        #for word in StopWords.StopWords: 
        #    reg_string = word + '-[a-zA-Z]*'
        #    data = re.sub(reg_string, '', data) 

        #get parts of speech
        tokens = nltk.word_tokenize(data)
        tagged_pos = nltk.pos_tag(tokens)
        filtered_pos = list(filter(lambda word_tag: word_tag[1] != 'NNP' and word_tag[1] != 'PRP' and word_tag[1] != 'PRP$', tagged_pos))

        #remove stop words
        punctuation_to_keep = "!.?-'"
        punctuation_to_remove = re.sub("([!.?'])", "", string.punctuation)     
        stop_words = StopWords.StopWords

        # with open('./debias/debiaswe/data/gender_specific_seed.json', "r") as f:
        #     gender_specific_words = json.load(f)

        #remove unwanted pos
        combined = "" 
        last_processed_punc = False
        
        for word, tag in filtered_pos:
            
            append = ''
            if (word[len(word) - 1] in string.punctuation and len(word) > 1):
                append = word[len(word) - 1]
                word = word.replace(word[len(word) - 1], '')

            if (not word.lower() in stop_words
                #and not word.lower() in gender_specific_words
                and not word in punctuation_to_remove
                and word != tag
                and (len(word) > 1 or word in punctuation_to_keep)):

                if (word in punctuation_to_keep):

                    if (last_processed_punc):
                        continue

                    combined += word + append
                    last_processed_punc = True
                else:
                    combined += " " + word + append               
                    last_processed_punc = False
             
        #stop_words = StopWords.StopWords            
        #whitespaces
        processed_data = combined.strip()
        #print(processed_data)
        
        return processed_data
    
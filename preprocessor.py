
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import unicodedata
import string 
import re
#from pycontractions import Contractions
import gensim.downloader as api
import StopWords

class Preprocessor():

   # def __init__(self):

       # self.cont = Contractions(api_key='glove-twitter-25')
       # self.cont.load_models()

    def Clean(self, data : str):
        ''' Removes POS that are NNP, PRP, or PRP$, and removes all stop words  '''

        #normalize the data, removing punctuation 
        data =  unicodedata.normalize('NFKC', data)
        
        #remove numbers
        data = re.sub('\d+', '', data)
   

        #expand contractions 
        #data = self.cont._expand_text_precise(data)[0]
      
        #get parts of speech
        tokens = nltk.word_tokenize(data)
        tagged_pos = nltk.pos_tag(tokens)
        filtered_pos = list(filter(lambda word_tag: word_tag[1] != 'NNP' and word_tag[1] != 'PRP' and word_tag[1] != 'PRP$', tagged_pos))
        #print(filtered_pos)
        #remove stop words
        punctuation_to_keep = "!.?-'"
        punctuation_to_remove = re.sub("([!.?'])", "", string.punctuation)     
        stop_words = StopWords.StopWords

        #remove unwanted pos
        combined = "" 
        last_processed_punc = False
        
        for word, tag in filtered_pos:
            if (not word.lower() in stop_words and not word in punctuation_to_remove and word != tag and len(word) > 1 or word in punctuation_to_keep):

                if (word in punctuation_to_keep):

                    if (last_processed_punc):
                        continue

                    combined += word 
                    last_processed_punc = True
                else:
                    combined += " " + word 
                    last_processed_punc = False
             
        #stop_words = StopWords.StopWords            
        #whitespaces
        processed_data = combined.strip()
        #print(processed_data)
        
        return processed_data
    
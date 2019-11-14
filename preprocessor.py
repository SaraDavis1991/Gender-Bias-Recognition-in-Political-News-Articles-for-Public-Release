
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import unicodedata
import string 
import re
from pycontractions import Contractions
import gensim.downloader as api

class Preprocessor():

    def __init__(self):

        self.cont = Contractions(api_key='glove-twitter-25')
        self.cont.load_models()

    def Clean(self, data : str, names_to_remove = None):
        ''' Removes POS that are NNP, PRP, or PRP$, and removes all stop words  '''

        #normalize the data, removing punctuation 
        data =  unicodedata.normalize('NFKC', data)
        punctuation_to_remove = re.sub("([!.?])", "", string.punctuation)

        #remove numbers
        data = re.sub('\d+', '', data)

        #expand contractions 
        data = self.cont._expand_text_precise(data)[0]

        #get parts of speech
        tokens = nltk.word_tokenize(data)
        tagged_pos = nltk.pos_tag(tokens)
        filtered_pos = list(filter(lambda word_tag: word_tag[1] != 'NNP' and word_tag[1] != 'PRP' and word_tag[1] != 'PRP$', tagged_pos))

        #remove stop words
        punctuation_to_remove = re.sub("([!.?])", "", string.punctuation)     
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [] # [w for word, _ in filtered_pos if not len(w) > 1] # not w in stop_words or not w in punctuation_to_remove or 

        #remove unwanted pos
        for (word, _) in filtered_pos:         
            if (not word in stop_words and not word in punctuation_to_remove):
                filtered_pos.append(word)

        #remove extraneous punctuation 

        #re-expand back into string
        combined = ' '.join(filtered_pos)
             

        #whitespaces
        processed_data = combined.strip()

        
        return processed_data
    

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import unicodedata
import string 

class Preprocessor():

    def Clean(self, data : str, names_to_remove = None):
        ''' Removes POS that are NNP, PRP, or PRP$, and removes all stop words  '''

        #normalize the data, removing punctuation 
        data = unicodedata.normalize('NFKC', data)
        data = data.translate(str.maketrans('', '', string.punctuation))

        #remove stop words
        tokens = nltk.word_tokenize(data)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if not w in stop_words and len(w) > 1] 

        #get parts of speech
        tagged_pos = nltk.pos_tag(filtered_tokens)
        filtered_pos = list(filter(lambda word_tag: word_tag[1] == 'NNP' or word_tag[1] == 'PRP' or word_tag[1] == 'PRP$', tagged_pos))

        #remove unwanted pos
        for (word, _) in filtered_pos:         
            if (word in filtered_tokens):
                filtered_tokens.remove(word) 

        #remove keywords 
        if (names_to_remove != None):
            for name in names_to_remove:
                if (name in filtered_tokens):
                    filtered_tokens.remove(name)  
             
        
        return ' '.join(filtered_tokens)
    
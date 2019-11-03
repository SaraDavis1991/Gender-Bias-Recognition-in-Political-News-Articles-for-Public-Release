
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

class Preprocessor():

    def Get_pos(self, data):
        tokens = nltk.word_tokenize(data)
        tagged_pos = nltk.pos_tag(tokens)
        return tagged_pos

    def Remove_pos(self, data : str, names_to_remove = None):

        data_split = data.split(' ')

        #get parts of speech
        pos = self.Get_pos(data)
        filtered_pos = list(filter(lambda word_tag: word_tag[1] == 'NNP' or word_tag[1] == 'PRP' or word_tag[1] == 'PRP$' , pos))

        #remove unwanted pos
        for (word, _) in filtered_pos:         
            if (word in data_split):
                data_split.remove(word)

        #remove keywords 
        if (names_to_remove != None):
            for name in names_to_remove:
                if (name in data_split):
                    data_split.remove(name)    
        
        data = ''
        return data.join(data_split)
    
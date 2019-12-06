
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
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

class Preprocessor():

    def Clean(self, data : str):
        ''' Removes POS that are NNP, PRP, or PRP$, and removes all stop words  '''

        #normalize the data, removing punctuation 
        data =  unicodedata.normalize('NFKC', data)
        
        #remove numbers
        data = re.sub('\d+', '', data)
        
        #remove "follow n on twitter"
        data = re.sub('(Follow)\s[a-zA-Z\s]*Twitter[a-zA-Z\s@]*.', '', data)

        nlp = en_core_web_sm.load()
        doc = nlp(data)
        
        for text, label in doc.ents:
            if label =='PERSON':
                text = 'they'
            elif label == 'GPE':
                text = 'place'
            

        print([(X.text, X.label_) for X in doc.ents])




        #remove whole words from stop list 
        #for word in StopWords.StopWords: 
        #    reg_string = word + '-[a-zA-Z]*'
        #    data = re.sub(reg_string, '', data) 
        '''
        #get parts of speech
        tokens = nltk.word_tokenize(data)
        tagged_pos = nltk.pos_tag(tokens)
        filtered_pos = list(filter(lambda word_tag: word_tag[1] != 'NNP' and word_tag[1] != 'PRP' and word_tag[1] != 'PRP$', tagged_pos))

        #remove stop words
        punctuation_to_keep = "!.?-'"
        punctuation_to_remove = re.sub("([!.?'])", "", string.punctuation)     
        stop_words = StopWords.StopWords

        with open('./debias/debiaswe/data/gender_specific_seed.json', "r") as f:
            gender_specific_words = json.load(f)

        #remove unwanted pos
        combined = "" 
        last_processed_punc = False
        
        for word, tag in filtered_pos:
            
            append = ''
            if (word[len(word) - 1] in string.punctuation and len(word) > 1):
                append = word[len(word) - 1]
                word = word.replace(word[len(word) - 1], '')

            if (not word.lower() in stop_words
                and not word.lower() in gender_specific_words
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
        '''


process = Preprocessor()
process.Clean("Ocasio-Cortez, who is a self-described 'democratic socialist,' made the comments in an interview published Tuesday. Yahoo! News reported: 'Are we headed to fascism? Yes. I don't think there's a question,' the congresswoman told Yahoo News hours after she 'toured' the detention facilities run by Customs and Border Protection. 'If you actually take the time to study, and to look at the steps, and to see how government transforms under authoritarian regimes, and look at the political decisions and patterns of this president, the answer is yes.' Last month Ocasio-Cortez sparked controversy when she described the migrant detention facilities as 'concentration camps on our southern border.' Her comment drew an immediate backlash from critics who accused her of trivializing Nazi concentration camps, while others, including some Holocaust survivors and scholars, said the comparison was a valid one. The freshman Democratic lawmaker from New York refused to back down from that comparison, and in her conversation with Yahoo News, Ocasio-Cortez argued that many things about President Trump echo that dark period in history. Read the full Yahoo! News article here. Also on Monday evening, National Border Patrol Council president Brandon Judd told Breitbart News Tonight that Ocasio-Cortez had lied about conditions at facilities where illegal aliens are being held near the border, as well as about the behavior of officers at the facilities. 'How can you have the moral high ground if you are going to throw facts out the window and spew falsehoods?', he commented. Ocasio-Cortez was elected in November as part of a Democratic Party victory in the U.S. House of Representatives, matching a pattern throughout recent U.S. history in which the opposition fares well in the midterm elections. Since taking office, Ocasio-Cortez has proposed ambitious new policies that grant sweeping powers to the government over the U.S. economy, envisioning a highly controlled system in which the state controls private industry and steers it toward overarching utopian goals, such as a shift to 100% renewable energy and the elimination of fossil fuels.")
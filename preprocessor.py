
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
import en_core_web_lg

class Preprocessor():
    
    def __init__(self):
                
        self.nlp = spacy.load("en_core_web_lg")

    def cleanup(self, token, lower = True):
        if lower:
            token = token = token.lower()
        return token.strip()

    def Clean(self, data : str):
        ''' Removes POS that are NNP, PRP, or PRP$, and removes all stop words  '''

        #normalize the data, removing punctuation 
        data =  unicodedata.normalize('NFKC', data)
        
        #remove numbers
        data = re.sub('\d+', '', data)
        
        #remove "follow n on twitter"
        data = re.sub('(Follow)\s[a-zA-Z\s]*Twitter[a-zA-Z\s@]*.', '', data)

        #remove any word containing woman replace with person

        doc = self.nlp(data)
      
        #Used tagged entities to replace names of places and people
        for ent in reversed(doc.ents):
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            if ent.label_ == 'PERSON' or ent.label_ == 'NORP' or ent.label_ == 'GPE' or ent.label == 'LOC':
                data = data[:ent.start_char] + ent.label_.lower() + data[ent.end_char:]


        #replace gendered words with person and non gendered pronouns
        data = re.sub('(?<![a-zA-Z])woman(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])women(?![a-zA-Z])', 'people', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])congresswoman(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])congresswomen(?![a-zA-Z])', 'people', data, flags = re.I)

        data = re.sub('(?<![a-zA-Z])man(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])men(?![a-zA-Z])', 'people', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])congressman(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])congressmen(?![a-zA-Z])', 'people', data, flags = re.I)

        data = re.sub('(?<![a-zA-Z])girl(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])boy(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])girls(?![a-zA-Z])', 'people', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])boys(?![a-zA-Z])', 'people', data, flags = re.I)

        data = re.sub('(?<![a-zA-Z])he(?![a-zA-Z])', 'they', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])she(?![a-zA-Z])', 'they', data, flags = re.I)

        data = re.sub('(?<![a-zA-Z])his(?![a-zA-Z])', 'their', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])hers(?![a-zA-Z])', 'theirs', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])her(?![a-zA-Z])', 'their', data, flags = re.I)

        data = re.sub( '(?<![a-zA-Z])female(?![a-zA-Z])', 'human', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])male(?![a-zA-Z])', ' human', data, flags  = re.I)
        data = re.sub('(?<![a-zA-Z])mother(?![a-zA-Z])', 'parent', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])mom(?![a-zA-Z])', 'parent', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])father(?![a-zA-Z])', 'parent', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])dad(?![a-zA-Z])', 'parent', data, flags = re.I)

        data = re.sub('(?<![a-zA-Z])Vladimir Putin(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Putin(?![a-zA-Z])', 'person', data, flags = re.I)

        data = re.sub('(?<![a-zA-Z])queen(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])king(?![a-zA-Z])', 'person', data, flags =re.I)
        
        #remove state tag and party tag. ie "(D-NV)"
        data = re.sub(r'(\([DRrd]-[a-zA-Z]+\))', '', data, flags =re.I)

        #remove twitter tags
        data = re.sub(r'(@[a-zA-Z_-]*)', '', data, flags =re.I)

        #In case a person's name was not tagged as a name, replace it with person

        data = re.sub('(?<![a-zA-Z])Obama(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Trump(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Mcconnell(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Biden(?![a-zA-Z])', 'person', data, flags = re.I) 
        data = re.sub('(?<![a-zA-Z])Sanders(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Barack Obama(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Michelle Obama(?![a-zA-Z])', 'person', data , flags = re.I)
        data = re.sub('(?<![a-zA-Z])Donald Trump(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Donald Trump Jr(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Melania Trump Jr(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Mitch Mcconnell(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Joe Biden(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Hunter Biden(?![a-zA-Z])', 'person', data, flags = re.I) 
        data = re.sub('(?<![a-zA-Z])Bernie Sanders(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Mitch Mcconnell(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Barack(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Michelle(?![a-zA-Z])', 'person', data , flags = re.I)
        data = re.sub('(?<![a-zA-Z])Donald(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Melania(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Mitch(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Joe(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Hunter(?![a-zA-Z])', 'person', data, flags = re.I) 
        data = re.sub('(?<![a-zA-Z])Bernie(?![a-zA-Z])', 'person', data, flags = re.I)

        #news outlets
        data = re.sub('(breitbart)', 'news source', data, flags = re.I)
        data = re.sub('(fox)', 'news source', data, flags = re.I)
        data = re.sub('(usa today)', 'news source', data, flags = re.I)
        data = re.sub('(huffpost)', 'news source', data, flags = re.I)
        data = re.sub('(new york times)', 'news source', data, flags = re.I)

        data = re.sub('(breitbart news)', 'news source', data, flags = re.I)
        data = re.sub('(fox news)', 'news source', data, flags = re.I)
        data = re.sub('(usa today news)', 'news source', data, flags = re.I)
        data = re.sub('(huffpost news)', 'news source', data, flags = re.I)
        data = re.sub('(huffington post)', 'news source', data, flags = re.I)
        data = re.sub('(new york times news)', 'news source', data, flags = re.I)
        data = re.sub('(nyt)', 'news source', data, flags = re.I)

        data = re.sub('((mrs.))', 'news source', data, flags = re.I)
        data = re.sub('((mr.))', 'news source', data, flags = re.I)
        

        data = re.sub('(?<![a-zA-Z])Ocasio-Cortez(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Ocasio Cortez(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Ocasio(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Cortez(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])AOC(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Clinton(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Warren(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Devos(?![a-zA-Z])', 'person', data, flags = re.I) 
        data = re.sub('(?<![a-zA-Z])Palin(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Alexandria Ocasio-Cortez(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Hillary Clinton(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Elizabeth Warren(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Betsy Devos(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Sarah Palin(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Alexandria(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Hillary(?![a-zA-Z])', 'person', data , flags = re.I)
        data = re.sub('(?<![a-zA-Z])Elizabeth(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Sarah(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])Betsy(?![a-zA-Z])', 'person', data, flags = re.I)

        #Remove these because they are closely associated with someone's office
        data = re.sub('(?<![a-zA-Z])secretary of state(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])secretary of education(?![a-zA-Z])', 'person', data, flags = re.I)

        data = re.sub('(?<![a-zA-Z])department of education(?![a-zA-Z])', 'organization', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])department of state(?![a-zA-Z])', 'organization', data, flags = re.I)

        data = re.sub('(?<![a-zA-Z])president of the united states of america(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])president of the united states(?![a-zA-Z])', 'person', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])president(?![a-zA-Z])', 'person', data, flags = re.I)
                
        data = re.sub('(?<![a-zA-Z])secretary(?![a-zA-Z])', '', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])s(?![a-zA-Z])', '', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])U.S.(?![a-zA-Z])', 'norp', data, flags = re.I)
        data = re.sub('(?<![a-zA-Z])person person(?![a-zA-Z])', 'person', data, flags = re.I)

        #Cleanup the punctuation
        data = re.sub(' ,', ',', data, flags = re.I)
        data = re.sub('  ', ' ', data, flags = re.I)

        #remove whole words from stop list 
        for word in StopWords.StopWords: 
            reg_string = word + '-[a-zA-Z]*'
            data = re.sub(reg_string, '', data) 
     
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
        '''
        #return processed_data
        return data
    

#process = Preprocessor()
#process.Clean("president Trump, Donald Trump, AOC, secretary Clinton, Trump. president")
#print(process.Clean("Ocasio-Cortez, who is a self-described 'democratic socialist,' made the comments in an interview published Tuesday. Yahoo! News reported: 'Are we headed to fascism? Yes. I don't think there's a question,' the congresswoman told Yahoo News hours after she 'toured' the detention facilities run by Customs and Border Protection. 'If you actually take the time to study, and to look at the steps, and to see how government transforms under authoritarian regimes, and look at the political decisions and patterns of this president, the answer is yes.' Last month Ocasio-Cortez sparked controversy when she described the migrant detention facilities as 'concentration camps on our southern border.' Her comment drew an immediate backlash from critics who accused her of trivializing Nazi concentration camps, while others, including some Holocaust survivors and scholars, said the comparison was a valid one. The freshman Democratic lawmaker from New York refused to back down from that comparison, and in her conversation with Yahoo News, Ocasio-Cortez argued that many things about President Trump echo that dark period in history. Read the full Yahoo! News article here. Also on Monday evening, National Border Patrol Council president Brandon Judd told Breitbart News Tonight that Ocasio-Cortez had lied about conditions at facilities where illegal aliens are being held near the border, as well as about the behavior of officers at the facilities. 'How can you have the moral high ground if you are going to throw facts out the window and spew falsehoods?', he commented. Ocasio-Cortez was elected in November as part of a Democratic Party victory in the U.S. House of Representatives, matching a pattern throughout recent U.S. history in which the opposition fares well in the midterm elections. Since taking office, Ocasio-Cortez has proposed ambitious new policies that grant sweeping powers to the government over the U.S. economy, envisioning a highly controlled system in which the state controls private industry and steers it toward overarching utopian goals, such as a shift to 100% renewable energy and the elimination of fossil fuels. Hers, his, she, he, Trump, shis, president Trump, president obama, secretary clinton, secretary of education, secretary of state, department of education, department of state person person "))
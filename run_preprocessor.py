#################################
# run_pretrain_and_finetune.py:
# This file will generate various versions (see below) of jsons after gathering them using links
#################################

from Orchestrator import *
import ApplicationConstants
from parse_sentences import run
from DataReader import *

orchestrator = Orchestrator()
reader = DataReader()


#Randomize the articles that were scraped from the links (UPDATE THIS ONCE WE HAVE A FILE WITH LINKS)
orchestrator.read_data(path = "Data/articles_v3.json", save=True,
                       savePath=ApplicationConstants.all_articles_random_v4, random=True, clean=False, number_of_articles=1000)

#Clean the randomized articles
orchestrator.read_data(path=ApplicationConstants.all_articles_random_v4, save=True,
                       savePath=ApplicationConstants.all_articles_random_v4_cleaned, random=True, clean=True, number_of_articles=1000)


#Create the candidate name sentence json using parse_sentences and the randomized (dirty) articles
run(0, ApplicationConstants.all_articles_random_v4_candidate_names)
#clean the candidate name json and save it
orchestrator.read_data(path = ApplicationConstants.all_articles_random_v4_candidate_names, save = True, savePath = ApplicationConstants.all_articles_random_v4_candidate_names_cleaned
                       , random = True, clean = True, number_of_articles=1000)

run(2, ApplicationConstants.all_articles_random_v4_candidate_names_len2)
#Clean the dirty sentences containing the candidate name, then do POS tagging, save ADJ
orchestrator.read_data(path=ApplicationConstants.all_articles_random_v4_candidate_names_len2, save=True,
                       savePath=ApplicationConstants.all_articles_random_v4_cleaned_pos_candidate_names, clean=True,
                        random=True, number_of_articles=1000, pos_tagged=True)

#Clean all of the sentences (regardless of if candidate name is there), then do POS tagging, save ADJ
#orchestrator.read_data(path=ApplicationConstants.all_articles_random_v4, save=True,
#                       savePath=ApplicationConstants.all_articles_random_v4_cleaned_pos, clean=True,
#                        random=True, number_of_articles = 1000, pos_tagged=True)
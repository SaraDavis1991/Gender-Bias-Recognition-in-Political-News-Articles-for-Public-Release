#################################
# run_BOW.py:
# This file will generate the different BOW findings.
#################################

from Orchestrator import Orchestrator

orchestrator = Orchestrator()

'''
Uncomment the option you wish to run; our "overall words" were obtained with  option 1, our "adjective words" were 
obtained with option 2. You can turn print_vocab off without affecting results. 
file_name_1 is the cumulative word vector for all words in the articles
file_name_2 is a vector of vectors for the words in each article
model_name is the name of the svm trained to create the BOW
not_pos is a bool that determines if the BOW is run on all words or adj. If True, all words
lemmad is a bool that determines if a lemma is applied to the words. if True, a lemma is applied
print_vocab is a bool that determines if all of the vocab from the BOW is printed- helps confirm proper cleaning occurs

NOTE: to run this file, articles must have been collected and run_preprocessor.py must have been run
'''

#OPTION 1: run run_bow on all words in vocab, without lemma, print vocab to confirm data is cleaned properly, on balanced article count per person
#orchestrator.run_bow( "BOW_models/np_cumulative_vec_ALLnoL_balanced.npy", "BOW_models/np_count_vec_ALLnoL_balanced.npy", "BOW_models/perceptron_ALLnoL_balanced",True, False, True, True) #notPos, lemmad, printvocab

#OPTION 2: run run_bow on adjectives in vocab, without lemma, print vocab to confirm proper cleaning, on balanced article count per person
#orchestrator.run_bow( "BOW_models/np_cumulative_vec_ADJnoL_balanced.npy", "BOW_models/np_count_vec_ADJnoL_balanced.npy", "BOW_models/perceptron_ADJnoL_balanced",False, False, True, True) #Pos, lemmad, printvocab

#OPTION 3: run run_bow on all words in vocab, without lemma, print vocab to confirm data is cleaned properly, on all articles (unbalanced)
orchestrator.run_bow( "BOW_models/np_cumulative_vec_ALLnoL_notbalanced.npy", "BOW_models/np_count_vec_ALLnoL_notbalanced.npy", "BOW_models/perceptron_ALLnoL_notbalanced",True, False, True, False) #notPos, lemmad, printvocab

#OPTION 4: run run_bow on adjectives in vocab, without lemma, print vocab to confirm proper cleaning, on all articles (unbalanced)
#orchestrator.run_bow( "BOW_models/np_cumulative_vec_ADJnoL_notbalanced.npy", "BOW_models/np_count_vec_ADJnoL_notbalanced.npy", "BOW_models/perceptron_ADJnoL_notbalanced",False, False, True, False) #Pos, lemmad, printvocab





                                 Gender-Bias-Recognition-in-Political-News-Articles is run using Python3 in the following way:

(1) Download the articles at the relevant links. We cannot provide them here, due to copyright concerns. Save the articles in .json format. Use our reader
options to randomize the articles, and save the new randomized json in the Data directory as "articles_random_v3.json". You can change the name, but if 
you do so, you will need to update run_preprocessor.py
(2) run run_preprocessor.py to generate the relevant .json files, including cleaned files, and adjective files

================================================================DOC2VEC EMBEDDING TESTS======================================================================
                                                       
(3a) Download all the news 2.0 from https://components.one/datasets/all-the-news-2-news-articles-dataset/ to the store directory
(3b) Run run_pretrain_and_finetune.py to replicate our doc2vec embedding tests. This file contains several options for pretraining and finetuning, including
an option that skips pretraining for comparison to a baseline. Simply uncomment the line that you wish to run.  The default runs the parameters shown in our
paper: cleaned all the news data, cleaned news bias data, pretraining enabled. All of the metrics are saved to the PretrainFinetuneMetrics directory.

=================================================================BAG OF WORDS TESTS==========================================================================

(4) Run run_BOW.py to replicate our bag of word results. Four options are provided in this file; uncomment the one you would like to replicate, or set a
different combination. To replicate the results found when run on all of the words, run OPTION 1 (the default). To replicate the results the results found
when run on just adjectives, run OPTION 2. If print_vocab = True, vocabulary can be found in the vocabulary directory.

=================================================================SENTIMENT ANALYSIS TESTS====================================================================
                                                    
(5a) Download the Hu and Liu 2004 sentiment lexicon from http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar, and remove 'trump', 'vice', and 
'right' to remain consistent with our cleaning methods. Save the newly cleaned file as "positive-words-notrump.txt" and "negative-words-notrump.txt"
in the same folder. 
(5b) Run run_sentiment.py



Graphs for ALL tests are saved to the visualizations directory.

Models for ALL tests are saved to the store directory.



==================================================================DATA CLEANING==============================================================================


We use a combination of Bolukbasi et al. (2016) and Zhao et al. (2018) to create our stopwords list. We remove the following terms, as well 


The following gendered words have been removed from the Zhao gendered word list due to gender ambiguity in America, their use as a verb, or their lack of 
usefulness to our application. 

Male words including their plural form:
"wizard"
"actor"
"host"
"governor"
"hero"
"Deer"
"bull"
"colt"
"Gelding"
"waiter"
"Sorcerer"
"barbershop"
"dude"
"salesman"
"god"
"lion"
female list (also removed "women" and "woman" forms):
"female_ejaculation"
"hair_salon"
"viagra"
"hen"
"doe"
"filly"
"mare"
"cow"


We add the following words:
"ms."
"misses"
"missus"
"mister"
"gynecologist"

A full list of all stopwords can be found in StopWords.py


Anything with "man" in the word  as the regex changes this to "person". I.e. -> Cameraman becomes "cameraperson". OUr full substitution list can be seen in
RegexSearchPatterns.py

=================================================================PACKAGES====================================================================================
TODO

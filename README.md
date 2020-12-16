<p align="center"> Gender-Bias-Recognition-in-Political-News-Articles is run using Python3 in the following way:
</p>

(1) Download the articles at the relevant links. We cannot provide them here, due to copyright concerns. Save the articles in .json format under the Data drector as "articles_random_v4.json". You can change the name, but if you do so, you will need to update run_preprocessor.py <br/>
(2) run run_preprocessor.py to generate the json files necessary for consequent tests. This will take a while: <br/>

(a) articles_random_v4.json - a randomized version of the articles in v3<br/>
(b) articles_random_v4_cleaned.json - a randomized version with our cleaning implemented (no gendered pronouns/personally identifiable information)<br/>
(c) articles_random_v4_sentences_candidate_names.json - a randomized version that only contains the sentences with the target's name in it<br/>
(d) articles_random_v4_sentences_candidate_names_cleaned.json - a randomized version that only contains the sentences with the target's name in it that has been cleaned (see above)<br/>
(e) articles_random_v4_sentences_len2.json - a randomized version that contains sentences with the target's name in it, where the article must be at least 2 sentences long<br/>
(f) articles_random_v4_cleaned_pos_candidate_names.json - a randomized version that contains articles with the adjectives in sentences containing the target's name<br/>

All files will be saved to the Data directory. <br/><br/>

<p align="center">DOC2VEC EMBEDDING TESTS </p>
                                                       
(3a) Download all the news 2.0 from https://components.one/datasets/all-the-news-2-news-articles-dataset/ to the store directory<br/>
(3b) Run run_pretrain_and_finetune.py to replicate our doc2vec embedding tests. This file contains several options for pretraining and finetuning, including
an option that skips pretraining for comparison to a baseline. Simply uncomment the line that you wish to run.  The default runs the parameters shown in our
paper: cleaned all the news data, cleaned news bias data, using .2 of all the news due to RAM constrains. All of the metrics are saved to the PretrainFinetuneMetrics directory.
TSNE visualizations are saved in the visualizations directory. <br/>

**NOTE**: We run a linear neural net model and validate it 10 times, selecting the best validation model before testing and averaging the best neural net results across five folds. Linear neural nets are not optimal, so the average results that you achieve **may** be slightly different than the results we report. 
 <br/>


<p align="center">BAG OF WORDS TESTS</p>

(4) Run run_BOW.py to replicate our bag of word results. Four options are provided in this file; uncomment the one you would like to replicate, or set a
different combination. To replicate the results found when run on all of the words, run OPTION 1 (the default). To replicate the results the results found
when run on just adjectives, run OPTION 2. If print_vocab = True, vocabulary can be found in the vocabulary directory. 
**note: these results are currently no bueno-- going to try ngrams next** 
Vocabulary is saved to the vocabulary directory. <br/>

<p align="center">SENTIMENT ANALYSIS TESTS</p>
                                                    
(5a) Download the Hu and Liu 2004 sentiment lexicon from http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
Remove 'vice' and change the blank line at the top of the file below the comments to a semicolon to remain consistent with our cleaning methods and save it as "negative-words-novice.txt" in the opinion-lexicon-English directory.
Remove 'trump', 'right',  and change the blank line at the top of the file below the comments to a semicolon to remain consistent with our cleaning methods and save it as "positive-words-notrump.txt" in the opinion-lexicon-English directory. <br/>
(5b) Run run_sentiment.py<br/>
Sentiment visualizations are saved in the results/visualizations/ directory <br/>

Models for ALL tests are saved to the store directory.



<p align="center">DATA CLEANING</p>


We use a combination of Bolukbasi et al. (2016) and Zhao et al. (2018) to create our stopwords list. We remove the following terms, as well 


The following gendered words have been removed from the Zhao gendered word list due to gender ambiguity in America, their use as a verb, or their lack of 
usefulness to our application. 

Male words including their plural form:<br/>
"wizard"<br/>
"actor"<br/>
"host"<br/>
"governor"<br/>
"hero"<br/>
"deer"<br/>
"bull"<br/>
"colt"<br/>
"gelding"<br/>
"waiter"<br/>
"sorcerer"<br/>
"barbershop"<br/>
"dude"<br/>
"salesman"<br/>
"god"<br/>
"lion"<br/> <br/>
female list (also removed "women" and "woman" forms):<br/><br/>
"female_ejaculation"<br/>
"hair_salon"<br/>
"viagra"<br/>
"hen"<br/>
"doe"<br/>
"filly"<br/>
"mare"<br/>
"cow"<br/>


We add the following words:<br/>
"ms."<br/>
"misses"<br/>
"missus"<br/>
"mister"<br/>
"gynecologist"<br/>

A full list of all stopwords can be found in StopWords.py<br/>


Anything with "man" in the word  as the regex changes this to "person". I.e. -> Cameraman becomes "cameraperson". Our full substitution list can be seen in
RegexSearchPatterns.py

<p align="center">PACKAGES</p>
We recommend using a conda environment for installs unless noted in "". We have tested everything in Ubuntu 18.04 and 20.04
python 3.7 (because of gensim)
scikit-learn 
nltk
"In a python terminal-> import nltk nltk.download('stopwords')"
spacy 
spacy en_core_web_lg "python3 -m spacy download en_core_web_lg"
matplotlib
gensim
smart_open 2.0.0
seaborn
"pip3 install python-interface"

<p align="center">OTHER NOTES</p>
There are still superfluous functions and files that Sara has not yet removed. We've got a lot of code that needs to be tracked before deletion, so I'm kind of scared to delete it -Sara

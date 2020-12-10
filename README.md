<p align="center"> Gender-Bias-Recognition-in-Political-News-Articles is run using Python3 in the following way:
</p>

(1) Download the articles at the relevant links. We cannot provide them here, due to copyright concerns. Save the articles in .json format. Use our reader
options to randomize the articles, and save the new randomized json in the Data directory as "articles_random_v3.json". You can change the name, but if 
you do so, you will need to update run_preprocessor.py <br/>
(2) run run_preprocessor.py to generate the relevant .json files, including cleaned files, and adjective files

<p align="center">DOC2VEC EMBEDDING TESTS </p>
                                                       
(3a) Download all the news 2.0 from https://components.one/datasets/all-the-news-2-news-articles-dataset/ to the store directory<br/>
(3b) Run run_pretrain_and_finetune.py to replicate our doc2vec embedding tests. This file contains the option to pretrain on dirty or clean all the news data and fine tune on dirty of clean news-bias data. Simply uncomment the line that you wish to run.  The default runs the parameters shown in our
paper: cleaned all the news data, cleaned news bias data. All of the metrics (precision, recall, f1) are saved to the metrics directory. All of the models are saved to the PretrainFinetuneStorage directory. All TSNE visualizations are saved to the visualizations directory. 

<p align="center">BAG OF WORDS TESTS</p>

(4) Run run_BOW.py to replicate our bag of word results. Four options are provided in this file; uncomment the one you would like to replicate, or set a
different combination. To replicate the results found when run on all of the words, run OPTION 1 (the default). To replicate the results the results found
when run on just adjectives, run OPTION 2. If print_vocab = True, vocabulary can be found in the vocabulary directory.

<p align="center">SENTIMENT ANALYSIS TESTS</p>
                                                    
(5a) Download the Hu and Liu 2004 sentiment lexicon from http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar, and remove 'trump', 'vice', and 
'right' to remain consistent with our cleaning methods. Save the newly cleaned file as "positive-words-notrump.txt" and "negative-words-novice.txt"
in the same folder that they were downloaded in. <br/>
(5b) Run run_sentiment.py<br/>



Graphs for ALL tests are saved to the visualizations directory.<br/>

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


Anything with "man" in the word  as the regex changes this to "person". I.e. -> Cameraman becomes "cameraperson". OUr full substitution list can be seen in
RegexSearchPatterns.py

<p align="center">PACKAGES</p>
TODO



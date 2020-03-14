3/14/2020

- articles_random_v2.json is the randomized version (vs randomizing every time, so as to not see differences in our results) of all the articles. This has the latest data with mitch mcconnell. This data is uncleaned. 

- articles_random_v2_clean.json is the cleaned version of articles_random_v2.json. It consisted of:
 - spacy POS replacement
 - regex replacement
 - stop word removal 

Additionally, this cleaned version has only 50 articles per candidate (or however much the source has for that candidate). 
 

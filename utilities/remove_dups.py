import os 
import json
from typing import List

old_article_path = '..\\Data\\articles.json'
new_article_path = '.\\newData\\Data\\articles.json'
output_file_path = '.\\newData\\Data\\articles_removed.json'
new_articles_never_seen = [] 
with open(old_article_path, 'r') as old_file:

    with open(new_article_path, 'r') as new_file: 

        old_data = json.load(old_file)
        new_data = json.load(new_file)

        for leaning in old_data: 

            old_articles = old_data[leaning]['articles']
            new_articles = new_data[leaning]['articles']

            old_titles = list(map(lambda old_article: old_article['title'], old_articles))
            new_articles_never_seen += list(filter(lambda article: article['title'] not in old_titles, new_articles))

with open(output_file_path, 'w') as output_file:

    json.dump(new_articles_never_seen, output_file, indent=3)
import csv
import json
import os.path
from os import path
import operator 

data = {}
url_index = {} 
json_already_added = []
input_file_path = ".\\Data\\candidate\\nyt_alexandria_ocasio-cortez.csv"
output_file_path = '.\\Data\\candidate\\nyt_alexandria_ocasio-cortez.revised.json'

def create_json(file):
    ''' Creates the json representation of the data
    '''
    data["articles"] = []
    index = 0

    #open .csv for reading        
    reader = csv.reader(file)
    fields = next(reader)
    #sort the rows by the scraper id to have the data in order   
    reader = sorted(reader, key=lambda row: int(row[0].replace('-', '')))

    for row in reader:
        article = {}
        url = row[3]
        #remove any artifacts that came from video links (so pages had both videos and articles linked on the same page)
        if ("video" in url):
            continue

        #there's duplicate records that are pulled and need to be combined into one      
        if (url in url_index):
            content = row[8]
            original_article_index = url_index[url]
            data["articles"][original_article_index]["content"] += " " + content
        else:    
            #setup data structures
            url_index[row[3]] = index

            article["title"] = row[4]
            article["url"] = row[3]
            article["subtitle"] = row[5]
            article["author"] = row[6]
            article["content"] = row[8] 
            article["date"] = row[7]
            article["labels"] = {                  
                "author_gender" : "",
                "target_gender" : "Female",
                "target_affiliation" : "Far_Left", 
                "target_name" : "Alexandria_Ocasio-Cortez"
            }

            data["articles"].append(article)

            index += 1

#open .csv and .json if it has already been created
#using latin1 since I originally created the file in python 2 
with open(input_file_path, 'r', encoding="UTF-8-sig") as file:   
    if (not path.exists(output_file_path)):
        create_json(file)
        
#write out the result
with open(output_file_path, 'w') as write_file:
    json.dump(data, write_file, indent=2)